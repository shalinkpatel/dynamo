use anyhow::{Context, Result};
use nixl_sys::{Agent as NixlAgent, AgentConfig as NixlAgentConfig, ThreadSync as NixlThreadSync};

const G2PB_NIXL_PROGRESS_THREAD_DELAY_US: u64 = 1_000;

pub fn g2pb_nixl_agent_name(worker_id: u64) -> String {
    worker_id.to_string()
}

/// Create a G2PB NIXL agent without installing any transfer backends.
///
/// Both the external host service and each physical TRT rank create a G2PB
/// agent. Keep their progress-thread policy identical so neither side busy
/// polls while the model is decoding without a host transfer in flight.
pub fn new_g2pb_nixl_agent(agent_name: &str) -> Result<NixlAgent> {
    NixlAgent::new_configured(
        agent_name,
        &g2pb_nixl_agent_config(G2PB_NIXL_PROGRESS_THREAD_DELAY_US),
    )
    .with_context(|| format!("failed to create G2PB NIXL agent {agent_name}"))
}

/// Same agent, with the progress-thread cadence left to the caller. HiCache G2
/// reads are bulk RMA on a dedicated prefetch thread, not decode-time traffic,
/// so they can afford a faster progress thread than a TRT worker can.
pub fn new_hicache_g2_nixl_agent(agent_name: &str, progress_delay_us: u64) -> Result<NixlAgent> {
    NixlAgent::new_configured(agent_name, &g2pb_nixl_agent_config(progress_delay_us))
        .with_context(|| format!("failed to create HiCache G2 NIXL agent {agent_name}"))
}

fn g2pb_nixl_agent_config(progress_delay_us: u64) -> NixlAgentConfig {
    NixlAgentConfig {
        // A zero delay makes UCX call poll(..., timeout=0) continuously even
        // with no transfers. UCX event FDs still wake this bounded wait early.
        pthr_delay_us: progress_delay_us,
        // Agent::new uses the C++ default synchronization mode; the Rust
        // AgentConfig default is None, so preserve the former behavior here.
        thread_sync: NixlThreadSync::Default,
        ..NixlAgentConfig::default()
    }
}

pub fn create_worker_ucx_backend(
    agent: &NixlAgent,
    g2_to_g2_enabled: bool,
    ucx_devices: Option<&str>,
) -> anyhow::Result<()> {
    let (_, mut ucx_params) = agent.get_plugin_params("UCX")?;
    if g2_to_g2_enabled {
        // Split each composite peer transfer across four dedicated UCX progress workers.
        // NIXL requires one additional shared worker for metadata and notifications.
        ucx_params.set("num_threads", "4")?;
        ucx_params.set("num_workers", "5")?;
        ucx_params.set("split_batch_size", "1")?;
        // Avoid the 8 KiB UCX default throttling large transfers on the TCP fallback.
        // An explicit UCX_TCP_TX_SEG_SIZE environment setting still takes precedence.
        ucx_params.set("engine_config", "TCP_TX_SEG_SIZE=64K")?;
    }
    if let Some(ucx_devices) = ucx_devices {
        // NIXL 1.4 advertises this option as `ucx_devices`, but its UCX engine
        // reads `device_list` and converts each entry to `<device>:1` before
        // setting UCX_NET_DEVICES. Use the consumed key so the topology split
        // is effective rather than silently leaving every HCA eligible.
        ucx_params.set("device_list", ucx_devices)?;
    }
    agent.create_backend("UCX", &ucx_params)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g2pb_nixl_progress_thread_does_not_busy_poll() {
        let config = g2pb_nixl_agent_config(G2PB_NIXL_PROGRESS_THREAD_DELAY_US);

        assert!(config.enable_prog_thread);
        assert_eq!(config.pthr_delay_us, 1_000);
        assert!(matches!(config.thread_sync, NixlThreadSync::Default));
        // The HiCache path may ask for a faster thread; everything else about
        // the agent must stay identical.
        let fast = g2pb_nixl_agent_config(0);
        assert_eq!(fast.pthr_delay_us, 0);
        assert_eq!(fast.enable_prog_thread, config.enable_prog_thread);
    }
}
