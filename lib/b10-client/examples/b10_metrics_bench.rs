//! CPU saturation probe for native recording, concurrent scraping, and request logging.
//! Usage: b10_metrics_bench <config.json> <threads> <requests/thread> <tokens> [off] [scrape]
use dynamo_b10_client::{GenerationMetrics, GenerationRequest, RequestContext, RouterRequestNew};
use dynamo_runtime::pipeline::context::Controller;
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 5,
        "expected config, threads, requests per thread, tokens per request"
    );
    let threads: usize = args[2].parse()?;
    let requests: usize = args[3].parse()?;
    let tokens: usize = args[4].parse()?;
    anyhow::ensure!(
        threads > 0 && requests > 0 && tokens > 0,
        "benchmark sizes must be positive"
    );
    let off = args.iter().any(|a| a == "off");
    let scrape = args.iter().any(|a| a == "scrape");
    let metrics = Arc::new(GenerationMetrics::new(serde_json::from_slice(
        &std::fs::read(&args[1])?,
    )?)?);
    let barrier = Arc::new(Barrier::new(threads + 1));
    let stopped = Arc::new(AtomicBool::new(false));
    let scraper = scrape.then(|| {
        let metrics = metrics.clone();
        let stopped = stopped.clone();
        std::thread::spawn(move || {
            let mut scrapes = 0;
            while !stopped.load(Ordering::Relaxed) {
                std::hint::black_box(metrics.samples());
                scrapes += 1;
                std::thread::sleep(Duration::from_millis(100));
            }
            scrapes
        })
    });
    let workers: Vec<_> = (0..threads).map(|index| {
        let metrics = metrics.clone(); let barrier = barrier.clone();
        std::thread::spawn(move || {
            let context = RequestContext::new(Arc::new(Controller::new(format!("org--request-{index}--version"))), None, BTreeMap::new());
            let request = GenerationRequest { routing_request: RouterRequestNew { tokens: vec![1; 256], ..Default::default() }, primary_worker_request: rmpv::Value::Nil, decode_worker_request: None };
            let chunk: rmpv::Value = serde_json::from_value(serde_json::json!({"outputs":[{"token_ids_diff":[10]}]})).unwrap();
            let terminal: rmpv::Value = serde_json::from_value(serde_json::json!({"finished":true,"cached_tokens":0,"outputs":[{"token_ids_diff":[],"finish_reason":"stop"}]})).unwrap();
            barrier.wait();
            for _ in 0..requests {
                let mut recorder = (!off).then(|| metrics.begin(&context, &request, 64).unwrap());
                for _ in 0..tokens {
                    std::hint::black_box(&chunk);
                    if let Some(recorder) = &mut recorder { recorder.chunk(&chunk); }
                }
                if let Some(recorder) = &mut recorder { recorder.chunk(&terminal); }
            }
        })
    }).collect();
    let start = Instant::now();
    barrier.wait();
    for worker in workers {
        worker.join().unwrap();
    }
    let elapsed = start.elapsed();
    stopped.store(true, Ordering::Relaxed);
    let scrapes = scraper.map(|s| s.join().unwrap()).unwrap_or(0);
    if !off {
        let samples = metrics.samples();
        let value = |name| {
            samples
                .iter()
                .filter(|s| s.name == name)
                .map(|s| s.value)
                .sum::<f64>()
        };
        assert_eq!(
            value("llm_output_tokens_total"),
            (threads * requests * tokens) as f64
        );
        assert_eq!(
            value("total_successful_requests"),
            (threads * requests) as f64
        );
        assert_eq!(value("current_inflight_requests"), 0.0);
    }
    println!(
        "{}",
        serde_json::json!({"threads":threads,"requests_per_thread":requests,"tokens_per_request":tokens,"recording":!off,"scrapes":scrapes,"elapsed_s":elapsed.as_secs_f64(),"ns_per_token":elapsed.as_nanos() as f64 / (threads*requests*tokens) as f64})
    );
    Ok(())
}
