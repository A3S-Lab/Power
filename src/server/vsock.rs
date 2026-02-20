//! Vsock transport: serve the Power HTTP API over AF_VSOCK for a3s-box
//! MicroVM guest-host communication.
//!
//! In a MicroVM, the host (a3s-box orchestrator) communicates with a3s-power
//! over a vsock socket rather than TCP, avoiding the need for any network
//! configuration inside the VM. The vsock server exposes the same axum router
//! as the plain HTTP and TLS listeners.
//!
//! This module is only compiled on Linux with the `vsock` feature enabled.

use tokio_vsock::{VsockAddr, VsockListener, VMADDR_CID_ANY};

use crate::error::{PowerError, Result};

/// Spawn a vsock HTTP server in a background task.
///
/// Binds on `VMADDR_CID_ANY` (accepts connections from any CID) at the
/// given port, then serves the axum router over AF_VSOCK.
///
/// Uses the same `axum::serve` path as the TCP listener — `tokio-vsock`
/// implements `axum::serve::Listener` for `VsockListener` via its `axum08`
/// feature, so the vsock server behaves identically to the TCP server.
pub async fn spawn_vsock_server(port: u32, app: axum::Router) -> Result<()> {
    let addr = VsockAddr::new(VMADDR_CID_ANY, port);
    let listener = VsockListener::bind(addr)
        .map_err(|e| PowerError::Server(format!("Failed to bind vsock port {port}: {e}")))?;

    tracing::info!(vsock_port = port, "Vsock server listening");

    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app.into_make_service()).await {
            tracing::error!("Vsock server error: {e}");
        }
    });

    Ok(())
}
