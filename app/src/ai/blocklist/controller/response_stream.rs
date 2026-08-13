use std::cell::RefCell;
#[cfg(all(feature = "tui", not(target_family = "wasm")))]
use std::collections::HashMap;
#[cfg(all(feature = "tui", not(target_family = "wasm")))]
use std::process::Stdio;
use std::rc::Rc;
use std::sync::Arc;
#[cfg(all(feature = "tui", not(target_family = "wasm")))]
use std::time::Duration;

use anyhow::anyhow;
use chrono::{DateTime, Local, TimeDelta};
#[cfg(all(feature = "tui", not(target_family = "wasm")))]
use futures::channel::mpsc;
use futures::channel::oneshot;
#[cfg(all(feature = "tui", not(target_family = "wasm")))]
use tokio::io::{
    AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader, BufWriter,
};
#[cfg(all(feature = "tui", not(target_family = "wasm")))]
use tokio::process::Command;
use uuid::Uuid;
use warp_errors::report_error;
#[cfg(not(target_family = "wasm"))]
use warp_multi_agent_api as maa_api;
use warp_multi_agent_api::response_event;
use warpui::{Entity, ModelContext, SingletonEntity};

use crate::ai::agent::api::{self, ConvertToAPITypeError, generate_multi_agent_output};
use crate::ai::agent::conversation::AIConversationId;
use crate::ai::agent::{AIIdentifiers, CancellationReason};
use crate::network::NetworkStatus;
use crate::send_telemetry_from_ctx;
use crate::server::server_api::{AIApiError, ServerApiProvider};

/// Maximum number of times a single MAA request is re-sent before the failure is
/// surfaced.
const MAX_RETRIES: usize = 3;

/// Maximum time to wait for a request-time Grok OAuth token refresh before
/// sending with the currently stored token. Bounded so a hung refresh can't
/// stall the request.
#[cfg(not(target_family = "wasm"))]
const GROK_REFRESH_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// How long a request will hold for a request-time GEAP credential mint before
/// giving up and sending anyway.
#[cfg(not(target_family = "wasm"))]
const GEAP_REFRESH_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// What to do about a failed or truncated MAA response attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryAction {
    /// Re-send the same request immediately.
    RetryNow,
    /// Re-send the same request once connectivity returns.
    RetryWhenOnline,
    /// Resume the conversation with a fresh request after the stream completes.
    Resume,
    /// Surface the error; the conversation ends in error.
    Fail,
}

/// Decides how to recover from a failed response-stream attempt.
///
/// Before any client actions have been received, the request can be re-sent verbatim
/// (immediately, or once connectivity returns). After actions have streamed,
/// re-sending is unsafe, so recovery uses a fresh `ResumeConversation` request.
fn recovery_action(
    has_received_client_actions: bool,
    is_recoverable: bool,
    has_retry_budget: bool,
    can_attempt_resume_on_error: bool,
    is_online: bool,
) -> RecoveryAction {
    if !has_received_client_actions && is_recoverable && has_retry_budget {
        if is_online {
            RecoveryAction::RetryNow
        } else {
            RecoveryAction::RetryWhenOnline
        }
    } else if has_received_client_actions && is_recoverable && can_attempt_resume_on_error {
        RecoveryAction::Resume
    } else {
        RecoveryAction::Fail
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResponseStreamId(String);

impl ResponseStreamId {
    pub fn for_shared_session(init_event: &response_event::StreamInit) -> Self {
        // Make the stream ID unique per viewing by appending a local UUID
        // This prevents collisions when replaying the same conversation multiple times
        // (either on close-and-reopen or when viewing the same shared session from multiple terminals)
        Self(format!("{}-{}", init_event.request_id, Uuid::new_v4()))
    }

    #[cfg(test)]
    pub fn new_for_test() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

/// Model wrapping an agent API response stream.
///
/// Emits events when the output corresponding to the stream is updated, typically after receiving
/// each response chunk.
///
/// Handles retries internally - retries are only attempted if no ClientActions events have been
/// received yet, ensuring we don't retry after the AI has started executing actions.
pub struct ResponseStream {
    id: ResponseStreamId,
    params: api::RequestParams,
    retry_count: usize,
    start_time: DateTime<Local>,
    time_to_latest_event: TimeDelta,
    cancellation_tx: Option<oneshot::Sender<()>>,
    /// Store the original error for telemetry when retries succeed
    original_error: Option<String>,
    /// Track whether we've received any client actions
    /// If true, we cannot retry on subsequent errors since actions may have been executed
    has_received_client_actions: bool,
    /// AI identifiers for telemetry emission
    ai_identifiers: AIIdentifiers,

    /// Whether this request can attempt to resume the conversation on error.
    /// This is true for all requests except those that are themselves the result of a resume
    /// triggered by a previous error.
    can_attempt_resume_on_error: bool,

    /// Whether we should attempt to resume the conversation after the stream finishes.
    ///
    /// This is set when a transient network/server failure occurs after client actions
    /// have been received (so an in-request retry is unsafe) and
    /// `can_attempt_resume_on_error` is true.
    should_resume_conversation_after_stream_finished: bool,

    /// Whether a `StreamFinished` event was received for the current request. A
    /// stream that completes without one was truncated in transit.
    stream_finished_received: bool,

    /// Whether a terminal error event has already been emitted for the current
    /// request, so stream completion doesn't synthesize a second failure for it.
    error_event_emitted: bool,

    /// Whether a retry is parked waiting for connectivity. While set, completion of
    /// the failed attempt's underlying stream is ignored.
    deferred_retry_pending: bool,

    /// Unique, internal id for the current request.
    ///
    /// This ensures that the model never emits events for a request that was already cancelled (or
    /// retried) and is still receiving lagging events.
    ///
    /// Note this is unique compared to `id`; this is unique across retry requests while the response
    /// stream id remains stable.
    current_request_id: Option<Uuid>,
}

impl ResponseStream {
    /// Emits a synthetic successful response event through the normal controller subscription.
    #[cfg(test)]
    pub fn emit_response_event_for_test(
        &mut self,
        event: warp_multi_agent_api::ResponseEvent,
        ctx: &mut ModelContext<Self>,
    ) {
        ctx.emit(ResponseStreamEvent::ReceivedEvent(Consumable::new(Ok(
            event,
        ))));
    }
    #[cfg(test)]
    pub fn new_for_test(id: ResponseStreamId) -> Self {
        let (cancellation_tx, _rx) = oneshot::channel();
        Self {
            id,
            params: api::RequestParams::new_for_test(),
            retry_count: 0,
            start_time: Local::now(),
            time_to_latest_event: TimeDelta::seconds(0),
            cancellation_tx: Some(cancellation_tx),
            original_error: None,
            has_received_client_actions: false,
            ai_identifiers: AIIdentifiers::default(),
            can_attempt_resume_on_error: false,
            should_resume_conversation_after_stream_finished: false,
            stream_finished_received: false,
            error_event_emitted: false,
            deferred_retry_pending: false,
            current_request_id: Some(Uuid::new_v4()),
        }
    }

    pub fn new(
        params: api::RequestParams,
        ai_identifiers: AIIdentifiers,
        can_attempt_resume_on_error: bool,
        ctx: &mut ModelContext<Self>,
    ) -> Self {
        let (cancellation_tx, cancellation_rx) = oneshot::channel();
        let start_time = Local::now();

        let request_id = Uuid::new_v4();
        Self::spawn_request(request_id, params.clone(), cancellation_rx, ctx);
        Self {
            id: ResponseStreamId(Uuid::new_v4().to_string()),
            params,
            start_time,
            time_to_latest_event: TimeDelta::seconds(0),
            cancellation_tx: Some(cancellation_tx),
            retry_count: 0,
            original_error: None,
            has_received_client_actions: false,
            ai_identifiers,
            can_attempt_resume_on_error,
            should_resume_conversation_after_stream_finished: false,
            stream_finished_received: false,
            error_event_emitted: false,
            deferred_retry_pending: false,
            current_request_id: Some(request_id),
        }
    }

    pub fn id(&self) -> &ResponseStreamId {
        &self.id
    }

    /// Returns true if we should attempt to resume the conversation after the stream finishes.
    pub fn should_resume_conversation_after_stream_finished(&self) -> bool {
        self.should_resume_conversation_after_stream_finished
    }

    /// Helper function to emit AgentModeError telemetry for error that is retryable (not user visible).
    fn emit_retryable_agent_mode_error_telemetry(
        &self,
        error: String,
        ctx: &mut ModelContext<Self>,
    ) {
        send_telemetry_from_ctx!(
            crate::TelemetryEvent::AgentModeError {
                identifiers: self.ai_identifiers.clone(),
                error,
                is_user_visible: false,
                will_attempt_to_resume: false,
            },
            ctx
        );
    }

    fn retry(&mut self, ctx: &mut ModelContext<Self>) {
        self.retry_count += 1;
        // Reset per-attempt state for the new attempt.
        self.has_received_client_actions = false;
        self.stream_finished_received = false;
        self.error_event_emitted = false;
        self.deferred_retry_pending = false;

        let (cancellation_tx, cancellation_rx) = oneshot::channel();
        if let Some(old_cancellation_tx) = self.cancellation_tx.take() {
            let _ = old_cancellation_tx.send(());
        }
        self.cancellation_tx = Some(cancellation_tx);

        let request_id = Uuid::new_v4();
        self.current_request_id = Some(request_id);
        Self::spawn_request(request_id, self.params.clone(), cancellation_rx, ctx);
    }

    /// Sends the request for `request_id`. When the request's model is served by
    /// the connected Grok subscription or may route to Gemini Enterprise, and
    /// that credential is already past hard expiry, this first blocks on a
    /// single shared refresh (owned by `ApiKeyManager`, so only one runs at a
    /// time) before sending. Requests with valid credentials, and requests for
    /// other providers, are sent directly.
    fn spawn_request(
        request_id: Uuid,
        params: api::RequestParams,
        cancellation_rx: oneshot::Receiver<()>,
        ctx: &mut ModelContext<Self>,
    ) {
        #[cfg(all(feature = "tui", not(target_family = "wasm")))]
        if crate::tui::tui_inference_provider() == crate::tui::TuiInferenceProvider::Codex {
            Self::spawn_codex_generate(request_id, params, cancellation_rx, ctx);
            return;
        }

        // The Grok subscription and its OAuth refresh are native-only.
        #[cfg(not(target_family = "wasm"))]
        {
            use ::ai::api_keys::{ApiKeyManager, GeapRefreshOutcome, GrokRefreshOutcome};
            use warpui::r#async::FutureExt as _;

            use crate::ai::llms::{LLMModelHost, LLMPreferences, LLMProvider};
            use crate::workspaces::user_workspaces::UserWorkspaces;

            // Only touch the Grok token for requests that actually use the Grok
            // subscription. The subscription is the only client-side source of
            // xAI auth (there's no BYO xAI key), so a base model whose provider
            // is xAI is exactly a subscription request.
            let uses_grok_subscription = LLMPreferences::as_ref(ctx)
                .get_llm_info(&params.model)
                .is_some_and(|info| info.provider == LLMProvider::Xai);
            if uses_grok_subscription {
                let byo_allowed = UserWorkspaces::as_ref(ctx).is_byo_api_key_enabled(ctx);
                // Reserve + start the shared refresh on `ApiKeyManager`'s context;
                // the in-flight guard is released there even if this stream is
                // dropped mid-refresh. `None` means the token is already usable.
                let refresh_rx = ApiKeyManager::handle(ctx).update(ctx, |manager, ctx| {
                    manager.begin_expired_grok_refresh(byo_allowed, ctx)
                });
                if let Some(refresh_rx) = refresh_rx {
                    let _ = ctx.spawn(
                        async move {
                            // Block on the shared refresh, bounded so a hung
                            // refresh can't stall the request forever.
                            refresh_rx.with_timeout(GROK_REFRESH_REQUEST_TIMEOUT).await
                        },
                        move |me, result, ctx| {
                            // Cancelled or superseded while refreshing — drop this attempt.
                            if me.current_request_id != Some(request_id) {
                                return;
                            }
                            if matches!(result, Ok(Ok(GrokRefreshOutcome::Refreshed))) {
                                // Send with the freshly refreshed token.
                                if let Some(access_token) = ApiKeyManager::as_ref(ctx)
                                    .grok_tokens()
                                    .and_then(|tokens| tokens.access_token_for_request())
                                    .map(str::to_owned)
                                    && let Some(keys) = me.params.api_keys.as_mut()
                                {
                                    keys.grok_oauth_access_token = access_token;
                                }
                                Self::spawn_generate(
                                    request_id,
                                    me.params.clone(),
                                    cancellation_rx,
                                    ctx,
                                );
                            } else {
                                // The refresh failed or timed out: don't send with
                                // the dead token — surface a terminal error asking
                                // the user to reconnect their subscription.
                                me.surface_grok_refresh_failure(request_id, ctx);
                            }
                        },
                    );
                    return;
                }
            }

            let uses_geap = LLMPreferences::as_ref(ctx)
                .get_llm_info(&params.model)
                .is_some_and(|info| {
                    info.host_configs
                        .get(&LLMModelHost::GeminiEnterprise)
                        .is_some_and(|host| host.enabled)
                });
            if uses_geap
                && let Some(binding) =
                    crate::ai::geap_credentials::current_geap_policy(ctx).mint_binding()
            {
                let refresh_binding = binding.clone();
                let refresh_rx = ApiKeyManager::handle(ctx).update(ctx, |manager, ctx| {
                    manager.begin_expired_geap_refresh(&binding, ctx, |manager, waiter, ctx| {
                        crate::ai::geap_credentials::start_geap_refresh_for_waiter(
                            manager, waiter, ctx,
                        );
                    })
                });
                if let Some(refresh_rx) = refresh_rx {
                    let _ = ctx.spawn(
                        async move { refresh_rx.with_timeout(GEAP_REFRESH_REQUEST_TIMEOUT).await },
                        move |me, result, ctx| {
                            // Cancelled or superseded while waiting — drop this attempt.
                            if me.current_request_id != Some(request_id) {
                                return;
                            }
                            // `RequestParams` snapshotted the credentials before
                            // the wait, so re-read just the GEAP credential and
                            // leave every other key alone.
                            //
                            // Unlike the Grok branch above, a mint failure, a
                            // timeout, or a dropped sender is never surfaced as a
                            // terminal error — the request goes out with the
                            // snapshot untouched, and it is the job of the server
                            // to respond with an error if the GEAP credentials are bad.
                            if matches!(result, Ok(Ok(GeapRefreshOutcome::Refreshed)))
                                && let Some(credentials) = ApiKeyManager::as_ref(ctx)
                                    .geap_credentials_for_request(&refresh_binding)
                            {
                                apply_geap_refresh_to_params(&mut me.params, Some(credentials));
                            }
                            Self::spawn_generate(
                                request_id,
                                me.params.clone(),
                                cancellation_rx,
                                ctx,
                            );
                        },
                    );
                    return;
                }
            }
        }

        Self::spawn_generate(request_id, params, cancellation_rx, ctx);
    }

    #[cfg(all(feature = "tui", not(target_family = "wasm")))]
    fn spawn_codex_generate(
        request_id: Uuid,
        params: api::RequestParams,
        cancellation_rx: oneshot::Receiver<()>,
        ctx: &mut ModelContext<Self>,
    ) {
        let _ = ctx.spawn(
            async move { codex_app_server_stream(params, cancellation_rx) },
            move |me, stream, ctx| {
                me.handle_response_stream_result(request_id, Ok(stream), ctx);
            },
        );
    }

    /// Emits a terminal, user-visible error for a failed request-time Grok token
    /// refresh instead of sending the request with an expired token. Mirrors the
    /// terminal-error emission in [`Self::handle_response_stream_result`].
    #[cfg(not(target_family = "wasm"))]
    fn surface_grok_refresh_failure(&mut self, request_id: Uuid, ctx: &mut ModelContext<Self>) {
        let error = Arc::new(AIApiError::GrokSubscriptionTokenRefreshFailed);
        self.error_event_emitted = true;
        self.report_request_failure(&error, NetworkStatus::as_ref(ctx).is_online());
        ctx.emit(ResponseStreamEvent::ReceivedEvent(Consumable::new(Err(
            error,
        ))));
        self.on_response_stream_complete(request_id, ctx);
    }

    /// Spawns the actual multi-agent request send for `request_id`.
    fn spawn_generate(
        request_id: Uuid,
        params: api::RequestParams,
        cancellation_rx: oneshot::Receiver<()>,
        ctx: &mut ModelContext<Self>,
    ) {
        let server_api = ServerApiProvider::as_ref(ctx).get();
        let _ = ctx.spawn(
            async move { generate_multi_agent_output(server_api, params, cancellation_rx).await },
            move |me, stream, ctx| {
                me.handle_response_stream_result(request_id, stream, ctx);
            },
        );
    }

    /// Cancels the stream. The conversation_id is preserved in the emitted event for async handling.
    pub(super) fn cancel(
        &mut self,
        reason: CancellationReason,
        conversation_id: AIConversationId,
        ctx: &mut ModelContext<Self>,
    ) {
        self.current_request_id = None;
        let Some(cancellation_tx) = self.cancellation_tx.take() else {
            return;
        };
        let _ = cancellation_tx.send(());
        ctx.emit(ResponseStreamEvent::AfterStreamFinished {
            cancellation: Some(StreamCancellation {
                reason,
                conversation_id,
            }),
        });
    }

    fn handle_response_stream_result(
        &mut self,
        request_id: Uuid,
        stream_result: Result<api::ResponseStream, ConvertToAPITypeError>,
        ctx: &mut ModelContext<Self>,
    ) {
        match stream_result {
            Ok(stream) => {
                ctx.spawn_stream_local(
                    stream,
                    move |me, event, ctx| {
                        me.handle_response_stream_event(request_id, event, ctx);
                    },
                    move |me, ctx| {
                        me.on_response_stream_complete(request_id, ctx);
                    },
                );
            }
            Err(e) => {
                report_error!(
                    anyhow::anyhow!("{e:?}").context("Failed to send request to multi-agent API")
                );
                if self.current_request_id.is_none_or(|id| id != request_id) {
                    return;
                }
                // A request-conversion failure is a deterministic client-side error and
                // no stream was ever created: retrying would fail identically, and
                // letting completion synthesize `UnexpectedEof` would misreport it as
                // a transient network failure. Surface the original error and finish
                // terminally. (HTTP send failures don't take this path — they arrive as
                // in-stream error events.)
                let error = Arc::new(AIApiError::Other(anyhow!(e)));
                self.error_event_emitted = true;
                self.report_request_failure(&error, NetworkStatus::as_ref(ctx).is_online());
                ctx.emit(ResponseStreamEvent::ReceivedEvent(Consumable::new(Err(
                    error,
                ))));
                self.on_response_stream_complete(request_id, ctx);
            }
        }
    }

    fn handle_response_stream_event(
        &mut self,
        request_id: Uuid,
        event: api::Event,
        ctx: &mut ModelContext<Self>,
    ) {
        if self.current_request_id.is_none_or(|id| id != request_id) {
            return;
        }
        self.time_to_latest_event = Local::now().signed_duration_since(self.start_time);

        match &event {
            Ok(response_event) => {
                if let Some(event_type) = &response_event.r#type {
                    match event_type {
                        warp_multi_agent_api::response_event::Type::Init(init_event) => {
                            // Capture server_output_id from StreamInit event
                            self.ai_identifiers.server_output_id =
                                Some(crate::ai::agent::ServerOutputId::new(
                                    init_event.request_id.clone(),
                                ));
                        }
                        warp_multi_agent_api::response_event::Type::ClientActions(_) => {
                            // Mark that we've received client actions
                            self.has_received_client_actions = true;
                        }
                        warp_multi_agent_api::response_event::Type::Finished(finished_event) => {
                            self.stream_finished_received = true;
                            // Emit retry success telemetry on successful completion
                            if matches!(
                                finished_event.reason,
                                Some(warp_multi_agent_api::response_event::stream_finished::Reason::Done(_)) | None
                            ) {
                                // Emit retry success telemetry if this was a successful completion after retries
                                if self.retry_count > 0
                                    && let Some(original_error) = &self.original_error {
                                        send_telemetry_from_ctx!(
                                            crate::TelemetryEvent::AgentModeRequestRetrySucceeded {
                                                identifiers: self.ai_identifiers.clone(),
                                                retry_count: self.retry_count,
                                                original_error: original_error.clone(),
                                            },
                                            ctx
                                        );
                                    }
                            }
                        }
                    }
                }
                ctx.emit(ResponseStreamEvent::ReceivedEvent(Consumable::new(event)));
            }
            Err(e) => {
                // Store original error if this is the first error
                if self.retry_count == 0 {
                    self.original_error = Some(format!("{e:?}"));
                }

                let is_online = NetworkStatus::as_ref(ctx).is_online();
                match recovery_action(
                    self.has_received_client_actions,
                    e.is_recoverable(),
                    self.retry_count < MAX_RETRIES,
                    self.can_attempt_resume_on_error,
                    is_online,
                ) {
                    RecoveryAction::RetryNow => {
                        log::warn!(
                            "MultiAgent request failed, retrying (attempt {}/{}) - Error: {e:?}",
                            self.retry_count + 1,
                            MAX_RETRIES
                        );
                        // Only emit error telemetry here if we're retrying.
                        // Final errors that aren't being retried are emitted elsewhere.
                        self.emit_retryable_agent_mode_error_telemetry(format!("{e:?}"), ctx);
                        self.retry(ctx);
                        // Don't emit the error event, we're retrying
                        return;
                    }
                    RecoveryAction::RetryWhenOnline => {
                        log::warn!(
                            "MultiAgent request failed while offline; retrying (attempt {}/{}) once connectivity returns - Error: {e:?}",
                            self.retry_count + 1,
                            MAX_RETRIES
                        );
                        self.emit_retryable_agent_mode_error_telemetry(format!("{e:?}"), ctx);
                        self.defer_retry_until_online(ctx);
                        return;
                    }
                    RecoveryAction::Resume => {
                        // Recoverable failure after client actions: we'll resume the
                        // conversation once the stream finishes rather than surface the
                        // error, so the UI suppresses the banner. Log it so the
                        // auto-recovery isn't completely silent.
                        log::warn!(
                            "MultiAgent request failed after client actions; resuming conversation after stream finishes - Error: {e:?}"
                        );
                        // The resume spawn itself waits for connectivity.
                        self.should_resume_conversation_after_stream_finished = true;
                    }
                    RecoveryAction::Fail => {}
                }
                self.error_event_emitted = true;

                self.report_request_failure(e, is_online);

                ctx.emit(ResponseStreamEvent::ReceivedEvent(Consumable::new(event)));
            }
        }
    }

    fn on_response_stream_complete(&mut self, request_id: Uuid, ctx: &mut ModelContext<Self>) {
        if self.current_request_id.is_none_or(|id| id != request_id) {
            return;
        }
        // A retry is parked waiting for connectivity; the request is logically still
        // active, so don't complete the stream for the failed attempt.
        if self.deferred_retry_pending {
            return;
        }

        // The server always sends a StreamFinished event before ending the response,
        // but a transport cut between chunks surfaces as a clean EOF. Synthesize the
        // failure and recover like any transient error.
        if !self.stream_finished_received && !self.error_event_emitted {
            log::warn!(
                "generate_multi_agent_output stream ended without emitting StreamFinished event."
            );
            let unexpected_eof = Arc::new(AIApiError::UnexpectedEof);
            let is_online = NetworkStatus::as_ref(ctx).is_online();
            match recovery_action(
                self.has_received_client_actions,
                unexpected_eof.is_recoverable(),
                self.retry_count < MAX_RETRIES,
                self.can_attempt_resume_on_error,
                is_online,
            ) {
                RecoveryAction::RetryNow => {
                    log::warn!(
                        "MultiAgent request failed, retrying (attempt {}/{}) - Error: {unexpected_eof:?}",
                        self.retry_count + 1,
                        MAX_RETRIES
                    );
                    self.emit_retryable_agent_mode_error_telemetry(
                        format!("{unexpected_eof:?}"),
                        ctx,
                    );
                    self.retry(ctx);
                    return;
                }
                RecoveryAction::RetryWhenOnline => {
                    log::warn!(
                        "MultiAgent request failed while offline; retrying (attempt {}/{}) once connectivity returns - Error: {unexpected_eof:?}",
                        self.retry_count + 1,
                        MAX_RETRIES
                    );
                    self.emit_retryable_agent_mode_error_telemetry(
                        format!("{unexpected_eof:?}"),
                        ctx,
                    );
                    self.defer_retry_until_online(ctx);
                    return;
                }
                RecoveryAction::Resume => {
                    // Recoverable truncation after client actions: we'll resume the
                    // conversation once the stream finishes rather than surface the
                    // error, so the UI suppresses the banner. Log it so the
                    // auto-recovery isn't completely silent.
                    log::warn!(
                        "MultiAgent request truncated after client actions; resuming conversation after stream finishes - Error: {unexpected_eof:?}"
                    );
                    self.should_resume_conversation_after_stream_finished = true;
                    self.error_event_emitted = true;
                    self.report_request_failure(&unexpected_eof, is_online);
                    ctx.emit(ResponseStreamEvent::ReceivedEvent(Consumable::new(Err(
                        unexpected_eof,
                    ))));
                }
                RecoveryAction::Fail => {
                    self.error_event_emitted = true;
                    self.report_request_failure(&unexpected_eof, is_online);
                    ctx.emit(ResponseStreamEvent::ReceivedEvent(Consumable::new(Err(
                        unexpected_eof,
                    ))));
                }
            }
        }

        ctx.emit(ResponseStreamEvent::AfterStreamFinished { cancellation: None });
        self.cancellation_tx = None;
    }

    /// Reports a non-retried request failure to crash reporting with classification
    /// tags.
    fn report_request_failure(&self, error: &Arc<AIApiError>, is_online: bool) {
        #[cfg(feature = "crash_reporting")]
        sentry::with_scope(
            |scope| {
                scope.set_tag(
                    "has_received_client_actions",
                    self.has_received_client_actions,
                );
                scope.set_tag("error", format!("{error:?}"));
                scope.set_tag("is_recoverable", error.is_recoverable());
                scope.set_tag(
                    "will_attempt_resume",
                    self.should_resume_conversation_after_stream_finished,
                );
                scope.set_tag("is_online", is_online);
            },
            || {
                report_error!(
                    error.as_ref(),
                    extra: {
                        "has_received_client_actions" => self.has_received_client_actions,
                        "is_recoverable" => error.is_recoverable(),
                        "will_attempt_resume" => self.should_resume_conversation_after_stream_finished,
                        "is_online" => is_online,
                        "retry_count" => self.retry_count,
                        "error_debug" => %format!("{error:?}"),
                    }
                );
            },
        );
        #[cfg(not(feature = "crash_reporting"))]
        {
            report_error!(
                error.as_ref(),
                extra: {
                    "has_received_client_actions" => self.has_received_client_actions,
                    "is_recoverable" => error.is_recoverable(),
                    "will_attempt_resume" => self.should_resume_conversation_after_stream_finished,
                    "is_online" => is_online,
                    "retry_count" => self.retry_count,
                    "error_debug" => %format!("{error:?}"),
                }
            );
        }
    }

    /// Parks a retry until connectivity returns; cancellation invalidates the parked
    /// retry through `current_request_id`.
    fn defer_retry_until_online(&mut self, ctx: &mut ModelContext<Self>) {
        self.deferred_retry_pending = true;
        ctx.emit(ResponseStreamEvent::WaitingForNetwork { waiting: true });
        let request_id_at_defer = self.current_request_id;
        let wait_for_online = NetworkStatus::as_ref(ctx).wait_until_online();
        let _ = ctx.spawn(wait_for_online, move |me, _, ctx| {
            // Cancelled or superseded while waiting — drop the parked retry.
            if request_id_at_defer.is_none() || me.current_request_id != request_id_at_defer {
                return;
            }
            ctx.emit(ResponseStreamEvent::WaitingForNetwork { waiting: false });
            me.retry(ctx);
        });
    }
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
const CODEX_INITIALIZE_REQUEST_ID: i64 = 0;
#[cfg(all(feature = "tui", not(target_family = "wasm")))]
const CODEX_THREAD_REQUEST_ID: i64 = 1;
#[cfg(all(feature = "tui", not(target_family = "wasm")))]
const CODEX_TURN_REQUEST_ID: i64 = 2;
#[cfg(all(feature = "tui", not(target_family = "wasm")))]
const CODEX_INTERRUPT_REQUEST_ID: i64 = 3;
#[cfg(all(feature = "tui", not(target_family = "wasm")))]
const CODEX_MAX_COMMAND_OUTPUT_BYTES: usize = 16 * 1024;

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
fn codex_app_server_stream(
    params: api::RequestParams,
    cancellation_rx: oneshot::Receiver<()>,
) -> api::ResponseStream {
    let (tx, rx) = mpsc::unbounded();
    tokio::spawn(async move {
        if let Err(error) = run_codex_app_server(params, cancellation_rx, tx.clone()).await {
            let _ = tx.unbounded_send(Err(Arc::new(AIApiError::Other(error))));
        }
    });
    Box::pin(rx)
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
async fn run_codex_app_server(
    params: api::RequestParams,
    mut cancellation_rx: oneshot::Receiver<()>,
    tx: mpsc::UnboundedSender<api::Event>,
) -> anyhow::Result<()> {
    let prompt = params
        .input
        .iter()
        .rev()
        .find_map(|input| input.user_query())
        .ok_or_else(|| anyhow!("Codex inference currently supports user prompts only"))?
        .to_owned();
    let (task_id, create_root_task) = params
        .tasks
        .first()
        .map(|task| (task.id.clone(), false))
        .unwrap_or_else(|| (Uuid::new_v4().to_string(), true));
    let existing_thread_id = params
        .conversation_token
        .as_ref()
        .map(|token| token.as_str().to_owned());
    let working_directory = params.session_context.current_working_directory();
    let cwd = working_directory.clone();

    let mut command = Command::new("codex");
    command
        .arg("app-server")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(working_directory) = &working_directory {
        command.current_dir(working_directory);
    }

    let mut child = command
        .spawn()
        .map_err(|error| anyhow!("Failed to launch Codex app-server: {error}"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("Failed to open Codex app-server stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("Failed to open Codex app-server stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("Failed to open Codex app-server stderr"))?;
    let stderr_task = tokio::spawn(async move {
        let mut stderr = stderr;
        let mut output = String::new();
        let _ = stderr.read_to_string(&mut output).await;
        output
    });
    let mut stdin = BufWriter::new(stdin);
    let mut lines = BufReader::new(stdout).lines();

    write_codex_rpc(
        &mut stdin,
        &serde_json::json!({
            "method": "initialize",
            "id": CODEX_INITIALIZE_REQUEST_ID,
            "params": {
                "clientInfo": {
                    "name": "warp_agent_cli",
                    "title": "Warp Agent CLI",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }
        }),
    )
    .await?;
    read_codex_rpc_response(
        &mut lines,
        CODEX_INITIALIZE_REQUEST_ID,
        &mut cancellation_rx,
    )
    .await?;
    write_codex_rpc(
        &mut stdin,
        &serde_json::json!({ "method": "initialized", "params": {} }),
    )
    .await?;

    let thread_request = if let Some(thread_id) = &existing_thread_id {
        serde_json::json!({
            "method": "thread/resume",
            "id": CODEX_THREAD_REQUEST_ID,
            "params": {
                "threadId": thread_id,
                "cwd": cwd,
                "approvalPolicy": "never",
                "sandbox": "workspace-write"
            }
        })
    } else {
        serde_json::json!({
            "method": "thread/start",
            "id": CODEX_THREAD_REQUEST_ID,
            "params": {
                "cwd": cwd,
                "approvalPolicy": "never",
                "sandbox": "workspace-write",
                "serviceName": "warp_agent_cli"
            }
        })
    };
    write_codex_rpc(&mut stdin, &thread_request).await?;
    let thread_result =
        read_codex_rpc_response(&mut lines, CODEX_THREAD_REQUEST_ID, &mut cancellation_rx).await?;
    let thread_id = thread_result
        .pointer("/thread/id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("Codex app-server did not return a thread ID"))?
        .to_owned();
    let mut mapper = CodexEventMapper::new(task_id, thread_id, create_root_task);
    send_codex_events(&tx, vec![Ok(mapper.init_event())])?;

    write_codex_rpc(
        &mut stdin,
        &serde_json::json!({
            "method": "turn/start",
            "id": CODEX_TURN_REQUEST_ID,
            "params": {
                "threadId": mapper.thread_id,
                "input": [{ "type": "text", "text": prompt }],
                "cwd": cwd,
                "approvalPolicy": "never",
                "sandboxPolicy": { "type": "workspaceWrite" }
            }
        }),
    )
    .await?;

    let mut completed = false;
    while !completed {
        tokio::select! {
            _ = &mut cancellation_rx => {
                if let Some(turn_id) = mapper.turn_id.as_deref() {
                    let _ = write_codex_rpc(
                        &mut stdin,
                        &serde_json::json!({
                            "method": "turn/interrupt",
                            "id": CODEX_INTERRUPT_REQUEST_ID,
                            "params": {
                                "threadId": mapper.thread_id,
                                "turnId": turn_id
                            }
                        }),
                    ).await;
                    tokio::time::sleep(Duration::from_millis(75)).await;
                }
                let _ = child.kill().await;
                return Ok(());
            }
            line = lines.next_line() => {
                let Some(line) = line.map_err(|error| anyhow!("Failed to read Codex app-server output: {error}"))? else {
                    break;
                };
                if line.trim().is_empty() {
                    continue;
                }
                let message: serde_json::Value = serde_json::from_str(&line)
                    .map_err(|error| anyhow!("Codex app-server returned invalid JSONL: {error}"))?;
                if message.get("id").and_then(serde_json::Value::as_i64)
                    == Some(CODEX_TURN_REQUEST_ID)
                {
                    if let Some(error) = message.get("error") {
                        return Err(codex_rpc_error(error));
                    }
                    if let Some(turn_id) = message
                        .pointer("/result/turn/id")
                        .and_then(serde_json::Value::as_str)
                    {
                        mapper.turn_id = Some(turn_id.to_owned());
                    }
                    continue;
                }
                let (events, turn_completed) = mapper.map_notification(&message)?;
                send_codex_events(&tx, events)?;
                completed = turn_completed;
            }
        }
    }

    if !completed {
        let status = child
            .wait()
            .await
            .map_err(|error| anyhow!("Failed while waiting for Codex app-server: {error}"))?;
        let stderr = stderr_task.await.unwrap_or_default();
        let detail = stderr.trim();
        return Err(if detail.is_empty() {
            anyhow!("Codex app-server exited before completing the turn ({status})")
        } else {
            anyhow!(detail.to_owned())
        });
    }

    let _ = child.kill().await;
    stderr_task.abort();
    Ok(())
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
async fn write_codex_rpc(
    stdin: &mut BufWriter<tokio::process::ChildStdin>,
    message: &serde_json::Value,
) -> anyhow::Result<()> {
    let mut line = serde_json::to_vec(message)
        .map_err(|error| anyhow!("Failed to encode Codex app-server request: {error}"))?;
    line.push(b'\n');
    stdin
        .write_all(&line)
        .await
        .map_err(|error| anyhow!("Failed to write to Codex app-server: {error}"))?;
    stdin
        .flush()
        .await
        .map_err(|error| anyhow!("Failed to flush Codex app-server request: {error}"))
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
async fn read_codex_rpc_response(
    lines: &mut tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    request_id: i64,
    cancellation_rx: &mut oneshot::Receiver<()>,
) -> anyhow::Result<serde_json::Value> {
    loop {
        let line = tokio::select! {
            _ = &mut *cancellation_rx => return Err(anyhow!("Codex request cancelled")),
            line = lines.next_line() => line
                .map_err(|error| anyhow!("Failed to read Codex app-server output: {error}"))?,
        };
        let Some(line) = line else {
            return Err(anyhow!("Codex app-server exited during initialization"));
        };
        if line.trim().is_empty() {
            continue;
        }
        let message: serde_json::Value = serde_json::from_str(&line)
            .map_err(|error| anyhow!("Codex app-server returned invalid JSONL: {error}"))?;
        if message.get("id").and_then(serde_json::Value::as_i64) != Some(request_id) {
            continue;
        }
        if let Some(error) = message.get("error") {
            return Err(codex_rpc_error(error));
        }
        return message
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow!("Codex app-server response did not include a result"));
    }
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
fn codex_rpc_error(error: &serde_json::Value) -> anyhow::Error {
    let message = error
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("Unknown Codex app-server error");
    let code = error.get("code").and_then(serde_json::Value::as_i64);
    match code {
        Some(code) => anyhow!("Codex app-server error {code}: {message}"),
        None => anyhow!(message.to_owned()),
    }
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
fn send_codex_events(
    tx: &mpsc::UnboundedSender<api::Event>,
    events: Vec<api::Event>,
) -> anyhow::Result<()> {
    for event in events {
        tx.unbounded_send(event)
            .map_err(|_| anyhow!("Codex response stream was closed"))?;
    }
    Ok(())
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodexMessageKind {
    AgentOutput,
    AgentReasoning,
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
impl CodexMessageKind {
    fn field_path(self) -> &'static str {
        match self {
            Self::AgentOutput => "agent_output.text",
            Self::AgentReasoning => "agent_reasoning.reasoning",
        }
    }
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
struct CodexEventMapper {
    task_id: String,
    thread_id: String,
    request_id: String,
    turn_id: Option<String>,
    create_root_task: bool,
    message_kinds: HashMap<String, CodexMessageKind>,
    command_output_bytes: HashMap<String, usize>,
    failure_message: Option<String>,
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
impl CodexEventMapper {
    fn new(task_id: String, thread_id: String, create_root_task: bool) -> Self {
        Self {
            task_id,
            thread_id,
            request_id: Uuid::new_v4().to_string(),
            turn_id: None,
            create_root_task,
            message_kinds: HashMap::new(),
            command_output_bytes: HashMap::new(),
            failure_message: None,
        }
    }

    fn init_event(&self) -> warp_multi_agent_api::ResponseEvent {
        warp_multi_agent_api::ResponseEvent {
            r#type: Some(response_event::Type::Init(response_event::StreamInit {
                request_id: self.request_id.clone(),
                conversation_id: self.thread_id.clone(),
                run_id: String::new(),
            })),
        }
    }

    fn map_notification(
        &mut self,
        message: &serde_json::Value,
    ) -> anyhow::Result<(Vec<api::Event>, bool)> {
        let Some(method) = message.get("method").and_then(serde_json::Value::as_str) else {
            return Ok((vec![], false));
        };
        let params = message.get("params").unwrap_or(&serde_json::Value::Null);
        let mut events = Vec::new();
        let mut completed = false;

        match method {
            "turn/started" => {
                self.turn_id = params
                    .pointer("/turn/id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
            }
            "item/agentMessage/delta" => {
                let (item_id, delta) = codex_delta(params)?;
                events.push(Ok(self.append_message(
                    item_id,
                    CodexMessageKind::AgentOutput,
                    delta,
                )));
            }
            "item/reasoning/summaryTextDelta" => {
                let (item_id, delta) = codex_delta(params)?;
                let summary_index = params
                    .get("summaryIndex")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or_default();
                events.push(Ok(self.append_message(
                    &format!("{item_id}-summary-{summary_index}"),
                    CodexMessageKind::AgentReasoning,
                    delta,
                )));
            }
            "item/commandExecution/outputDelta" => {
                let (item_id, delta) = codex_delta(params)?;
                if let Some(delta) = self.command_output_delta(item_id, delta) {
                    events.push(Ok(self.append_message(
                        item_id,
                        CodexMessageKind::AgentReasoning,
                        &delta,
                    )));
                }
            }
            "item/started" => {
                if let Some(event) = self.map_item_started(params)? {
                    events.push(Ok(event));
                }
            }
            "item/completed" => {
                if let Some(event) = self.map_item_completed(params)? {
                    events.push(Ok(event));
                }
            }
            "turn/plan/updated" => {
                if let Some(event) = self.map_plan(params) {
                    events.push(Ok(event));
                }
            }
            "error" => {
                self.failure_message = params
                    .pointer("/error/message")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
            }
            "turn/completed" => {
                for item in params
                    .pointer("/turn/items")
                    .and_then(serde_json::Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if item.get("type").and_then(serde_json::Value::as_str) == Some("agentMessage")
                        && let (Some(item_id), Some(text)) = (
                            item.get("id").and_then(serde_json::Value::as_str),
                            item.get("text").and_then(serde_json::Value::as_str),
                        )
                    {
                        events.push(Ok(self.set_message(
                            item_id,
                            CodexMessageKind::AgentOutput,
                            text,
                        )));
                    }
                }
                let status = params
                    .pointer("/turn/status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("failed");
                match status {
                    "completed" => events.push(Ok(self.finished_event(true))),
                    "interrupted" => events.push(Ok(self.finished_event(false))),
                    _ => {
                        let message = params
                            .pointer("/turn/error/message")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_owned)
                            .or_else(|| self.failure_message.take())
                            .unwrap_or_else(|| "Codex turn failed".to_owned());
                        events.push(Err(Arc::new(AIApiError::Other(anyhow!(message)))));
                    }
                }
                completed = true;
            }
            "warning"
            | "configWarning"
            | "thread/started"
            | "thread/status/changed"
            | "thread/tokenUsage/updated"
            | "turn/diff/updated"
            | "item/plan/delta"
            | "item/reasoning/summaryPartAdded"
            | "item/reasoning/textDelta"
            | "item/fileChange/outputDelta" => {}
            _ => {}
        }

        Ok((events, completed))
    }

    fn map_item_started(
        &mut self,
        params: &serde_json::Value,
    ) -> anyhow::Result<Option<warp_multi_agent_api::ResponseEvent>> {
        let item = params
            .get("item")
            .ok_or_else(|| anyhow!("Codex item/started notification omitted the item"))?;
        let Some(item_type) = item.get("type").and_then(serde_json::Value::as_str) else {
            return Ok(None);
        };
        let Some(item_id) = item.get("id").and_then(serde_json::Value::as_str) else {
            return Ok(None);
        };
        let text = match item_type {
            "commandExecution" => {
                let command = item
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("command");
                format!("Running:\n```sh\n{command}\n```\n")
            }
            "fileChange" => "Applying file changes…".to_owned(),
            "mcpToolCall" => format!(
                "Calling MCP tool `{}/{}`…",
                item.get("server")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("server"),
                item.get("tool")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("tool")
            ),
            "dynamicToolCall" => format!(
                "Running tool `{}`…",
                item.get("tool")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("tool")
            ),
            "collabAgentToolCall" => format!(
                "Coordinating agents with `{}`…",
                item.get("tool")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("agent tool")
            ),
            "webSearch" => "Searching the web…".to_owned(),
            _ => return Ok(None),
        };
        Ok(Some(self.set_message(
            item_id,
            CodexMessageKind::AgentReasoning,
            &text,
        )))
    }

    fn map_item_completed(
        &mut self,
        params: &serde_json::Value,
    ) -> anyhow::Result<Option<warp_multi_agent_api::ResponseEvent>> {
        let item = params
            .get("item")
            .ok_or_else(|| anyhow!("Codex item/completed notification omitted the item"))?;
        let Some(item_type) = item.get("type").and_then(serde_json::Value::as_str) else {
            return Ok(None);
        };
        let Some(item_id) = item.get("id").and_then(serde_json::Value::as_str) else {
            return Ok(None);
        };
        let (kind, text) = match item_type {
            "agentMessage" => (
                CodexMessageKind::AgentOutput,
                item.get("text")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            ),
            "commandExecution" => {
                let command = item
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("command");
                let exit = item
                    .get("exitCode")
                    .and_then(serde_json::Value::as_i64)
                    .map(|code| format!("exit {code}"))
                    .unwrap_or_else(|| {
                        item.get("status")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("completed")
                            .to_owned()
                    });
                let output = item
                    .get("aggregatedOutput")
                    .and_then(serde_json::Value::as_str)
                    .map(|output| truncate_codex_output(output, CODEX_MAX_COMMAND_OUTPUT_BYTES))
                    .filter(|output| !output.is_empty())
                    .map(|output| format!("\n```text\n{output}\n```"))
                    .unwrap_or_default();
                (
                    CodexMessageKind::AgentReasoning,
                    format!("Ran:\n```sh\n{command}\n```\n{exit}.{output}"),
                )
            }
            "fileChange" => {
                let count = item
                    .get("changes")
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, Vec::len);
                let status = item
                    .get("status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("completed");
                (
                    CodexMessageKind::AgentReasoning,
                    format!("File changes: {status} ({count} updates)."),
                )
            }
            "mcpToolCall" => (
                CodexMessageKind::AgentReasoning,
                format!(
                    "MCP tool `{}/{}`: {}.",
                    item.get("server")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("server"),
                    item.get("tool")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("tool"),
                    item.get("status")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("completed")
                ),
            ),
            "dynamicToolCall" | "collabAgentToolCall" => (
                CodexMessageKind::AgentReasoning,
                format!(
                    "Tool `{}`: {}.",
                    item.get("tool")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("tool"),
                    item.get("status")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("completed")
                ),
            ),
            "webSearch" => (
                CodexMessageKind::AgentReasoning,
                "Web search completed.".to_owned(),
            ),
            _ => return Ok(None),
        };
        Ok(Some(self.set_message(item_id, kind, &text)))
    }

    fn map_plan(
        &mut self,
        params: &serde_json::Value,
    ) -> Option<warp_multi_agent_api::ResponseEvent> {
        let turn_id = params
            .get("turnId")
            .and_then(serde_json::Value::as_str)
            .or(self.turn_id.as_deref())?;
        let entries = params.get("plan")?.as_array()?;
        let mut text = String::from("Plan:\n");
        for entry in entries {
            let step = entry
                .get("step")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let status = entry
                .get("status")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("pending");
            text.push_str(&format!("- [{status}] {step}\n"));
        }
        Some(self.set_message(
            &format!("codex-plan-{turn_id}"),
            CodexMessageKind::AgentReasoning,
            &text,
        ))
    }

    fn append_message(
        &mut self,
        message_id: &str,
        kind: CodexMessageKind,
        delta: &str,
    ) -> warp_multi_agent_api::ResponseEvent {
        if !self.message_kinds.contains_key(message_id) {
            return self.add_message(message_id, kind, delta);
        }
        use warp_multi_agent_api::client_action;

        self.client_actions_event(vec![warp_multi_agent_api::ClientAction {
            action: Some(client_action::Action::AppendToMessageContent(
                client_action::AppendToMessageContent {
                    task_id: self.task_id.clone(),
                    message: Some(self.message(message_id, kind, delta)),
                    mask: Some(prost_types::FieldMask {
                        paths: vec![kind.field_path().to_owned()],
                    }),
                },
            )),
        }])
    }

    fn set_message(
        &mut self,
        message_id: &str,
        kind: CodexMessageKind,
        text: &str,
    ) -> warp_multi_agent_api::ResponseEvent {
        if !self.message_kinds.contains_key(message_id) {
            return self.add_message(message_id, kind, text);
        }
        use warp_multi_agent_api::client_action;

        self.client_actions_event(vec![warp_multi_agent_api::ClientAction {
            action: Some(client_action::Action::UpdateTaskMessage(
                client_action::UpdateTaskMessage {
                    task_id: self.task_id.clone(),
                    message: Some(self.message(message_id, kind, text)),
                    mask: Some(prost_types::FieldMask {
                        paths: vec![kind.field_path().to_owned()],
                    }),
                },
            )),
        }])
    }

    fn add_message(
        &mut self,
        message_id: &str,
        kind: CodexMessageKind,
        text: &str,
    ) -> warp_multi_agent_api::ResponseEvent {
        use warp_multi_agent_api::client_action;

        self.message_kinds.insert(message_id.to_owned(), kind);
        let mut actions = Vec::with_capacity(if self.create_root_task { 2 } else { 1 });
        if self.create_root_task {
            self.create_root_task = false;
            actions.push(warp_multi_agent_api::ClientAction {
                action: Some(client_action::Action::CreateTask(
                    client_action::CreateTask {
                        task: Some(warp_multi_agent_api::Task {
                            id: self.task_id.clone(),
                            messages: vec![],
                            dependencies: None,
                            description: String::new(),
                            summary: String::new(),
                            server_data: String::new(),
                        }),
                    },
                )),
            });
        }
        actions.push(warp_multi_agent_api::ClientAction {
            action: Some(client_action::Action::AddMessagesToTask(
                client_action::AddMessagesToTask {
                    task_id: self.task_id.clone(),
                    messages: vec![self.message(message_id, kind, text)],
                },
            )),
        });
        self.client_actions_event(actions)
    }

    fn message(
        &self,
        message_id: &str,
        kind: CodexMessageKind,
        text: &str,
    ) -> warp_multi_agent_api::Message {
        let message = match kind {
            CodexMessageKind::AgentOutput => warp_multi_agent_api::message::Message::AgentOutput(
                warp_multi_agent_api::message::AgentOutput {
                    text: text.to_owned(),
                },
            ),
            CodexMessageKind::AgentReasoning => {
                warp_multi_agent_api::message::Message::AgentReasoning(
                    warp_multi_agent_api::message::AgentReasoning {
                        reasoning: text.to_owned(),
                        finished_duration: None,
                    },
                )
            }
        };
        warp_multi_agent_api::Message {
            fetched_memories: vec![],
            id: message_id.to_owned(),
            task_id: self.task_id.clone(),
            request_id: self.request_id.clone(),
            timestamp: None,
            server_message_data: String::new(),
            citations: vec![],
            message: Some(message),
        }
    }

    fn client_actions_event(
        &self,
        actions: Vec<warp_multi_agent_api::ClientAction>,
    ) -> warp_multi_agent_api::ResponseEvent {
        warp_multi_agent_api::ResponseEvent {
            r#type: Some(response_event::Type::ClientActions(
                response_event::ClientActions { actions },
            )),
        }
    }

    fn finished_event(&self, done: bool) -> warp_multi_agent_api::ResponseEvent {
        use warp_multi_agent_api::response_event::stream_finished;

        let reason = if done {
            stream_finished::Reason::Done(stream_finished::Done {})
        } else {
            stream_finished::Reason::Other(stream_finished::Other {})
        };
        warp_multi_agent_api::ResponseEvent {
            r#type: Some(response_event::Type::Finished(
                response_event::StreamFinished {
                    reason: Some(reason),
                    conversation_usage_metadata: None,
                    token_usage: vec![],
                    should_refresh_model_config: false,
                    request_cost: None,
                },
            )),
        }
    }

    fn command_output_delta(&mut self, item_id: &str, delta: &str) -> Option<String> {
        let bytes = self
            .command_output_bytes
            .entry(item_id.to_owned())
            .or_default();
        if *bytes >= CODEX_MAX_COMMAND_OUTPUT_BYTES {
            return None;
        }
        let remaining = CODEX_MAX_COMMAND_OUTPUT_BYTES - *bytes;
        let truncated = truncate_codex_output(delta, remaining);
        *bytes += truncated.len();
        let was_truncated = truncated.len() < delta.len();
        if was_truncated {
            *bytes = CODEX_MAX_COMMAND_OUTPUT_BYTES;
            Some(format!("{truncated}\n…command output truncated…\n"))
        } else {
            Some(truncated)
        }
    }
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
fn codex_delta(params: &serde_json::Value) -> anyhow::Result<(&str, &str)> {
    let item_id = params
        .get("itemId")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("Codex delta notification omitted itemId"))?;
    let delta = params
        .get("delta")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("Codex delta notification omitted delta"))?;
    Ok((item_id, delta))
}

#[cfg(all(feature = "tui", not(target_family = "wasm")))]
fn truncate_codex_output(output: &str, max_bytes: usize) -> String {
    if output.len() <= max_bytes {
        return output.to_owned();
    }
    let mut end = max_bytes;
    while !output.is_char_boundary(end) {
        end -= 1;
    }
    output[..end].to_owned()
}

/// Applies the result of a request-time GEAP mint to the request snapshot.
///
/// A successful mint swaps in the fresh credential.
#[cfg(not(target_family = "wasm"))]
fn apply_geap_refresh_to_params(
    params: &mut api::RequestParams,
    fresh_credentials: Option<maa_api::request::settings::api_keys::GoogleCloudCredentials>,
) {
    if let Some(credentials) = fresh_credentials
        && let Some(keys) = params.api_keys.as_mut()
    {
        keys.google_cloud_credentials = Some(credentials);
    }
}

#[derive(Debug)]
pub struct Consumable<T> {
    value: Rc<RefCell<Option<T>>>,
}

impl<T> Consumable<T> {
    fn new(value: T) -> Self {
        Consumable {
            value: Rc::new(RefCell::new(Some(value))),
        }
    }

    pub(super) fn consume(&self) -> Option<T> {
        self.value.borrow_mut().take()
    }
}

impl<T> Clone for Consumable<T> {
    fn clone(&self) -> Self {
        Consumable {
            value: Rc::clone(&self.value),
        }
    }
}

/// Cancellation context preserved for async event handling.
/// Includes conversation_id because truncation can remove exchange mappings before the event is processed.
#[derive(Debug, Clone)]
pub struct StreamCancellation {
    pub reason: CancellationReason,
    pub conversation_id: AIConversationId,
}

#[derive(Debug, Clone)]
pub enum ResponseStreamEvent {
    ReceivedEvent(Consumable<api::Event>),
    /// A retry is parked until connectivity returns (`waiting: true`) or has just
    /// fired (`waiting: false`). The controller mirrors this on the conversation
    /// status (`TransientError` ↔ `InProgress`).
    ///
    /// Only emitted from `defer_retry_until_online`, i.e. always after a recoverable
    /// request failure while offline — never speculatively before an attempt. Consumers
    /// can therefore treat `waiting: true` as a transient-error (reconnecting) state.
    WaitingForNetwork {
        waiting: bool,
    },
    AfterStreamFinished {
        /// Some for cancellation (with context), None for natural completion (uses dynamic lookup).
        cancellation: Option<StreamCancellation>,
    },
}

impl Entity for ResponseStream {
    type Event = ResponseStreamEvent;
}

#[cfg(test)]
#[path = "response_stream_tests.rs"]
mod tests;
