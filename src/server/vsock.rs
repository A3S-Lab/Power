//! Vsock transport: serve the Power HTTP API over AF_VSOCK for a3s-box
//! MicroVM guest-host communication.
//!
//! In a MicroVM, the host (a3s-box orchestrator) communicates with a3s-power
//! over a vsock socket rather than TCP, avoiding the need for any network
//! configuration inside the VM. The vsock server exposes the same axum router
//! as the plain HTTP and TLS listeners.
//!
//! This module is only used on Linux with the `vsock` feature enabled.

use std::io;
use std::time::Duration;

use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use hyper_util::service::TowerToHyperService;
use tokio_vsock::{VsockAddr, VsockListener, VMADDR_CID_ANY};

use crate::error::{PowerError, Result};

/// Spawn a vsock HTTP server in a background task.
///
/// Binds on `VMADDR_CID_ANY` (accepts connections from any CID) at the
/// given port, then serves the axum router over AF_VSOCK.
pub async fn spawn_vsock_server(port: u32, app: axum::Router) -> Result<()> {
    let addr = VsockAddr::new(VMADDR_CID_ANY, port);
    let listener = VsockListener::bind(addr)
        .map_err(|e| PowerError::Server(format!("Failed to bind vsock port {port}: {e}")))?;

    tracing::info!(vsock_port = port, "Vsock server listening");

    tokio::spawn(async move {
        serve_vsock(listener, app).await;
    });

    Ok(())
}

async fn serve_vsock(listener: VsockListener, app: axum::Router) {
    loop {
        let (stream, peer_addr) = match listener.accept().await {
            Ok(conn) => conn,
            Err(err) => {
                handle_accept_error(err).await;
                continue;
            }
        };

        let io = TokioIo::new(stream);
        let service = TowerToHyperService::new(app.clone());

        tokio::spawn(async move {
            if let Err(err) = Builder::new(TokioExecutor::new())
                .serve_connection_with_upgrades(io, service)
                .await
            {
                tracing::debug!(
                    peer = ?peer_addr,
                    error = %err,
                    "Vsock connection closed with error"
                );
            }
        });
    }
}

async fn handle_accept_error(err: io::Error) {
    if matches!(
        err.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    ) {
        return;
    }

    tracing::error!(error = %err, "Vsock accept error");
    tokio::time::sleep(Duration::from_secs(1)).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_entrypoint_compiles_for_feature_builds() {
        let _spawn = spawn_vsock_server;
    }
}
