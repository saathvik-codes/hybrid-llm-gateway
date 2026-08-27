import httpx
import time
import asyncio
import sys
import argparse

# Color formatting for terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

GATEWAY_URL = "http://127.0.0.1:8000"

# Simulation workload for automated benchmark
WORKLOAD = [
    # 1. Simple queries (Expected: local_slm)
    {"prompt": "Hello! How are you doing today?", "expected_route": "local_slm"},
    {"prompt": "What is the capital of Japan?", "expected_route": "local_slm"},
    
    # 2. Semantic Cache Hits (Expected: cache - close match to query 2)
    {"prompt": "Tell me the capital of Japan.", "expected_route": "cache"},
    
    # 3. Complex Reasoning queries (Expected: cloud_llm)
    {
        "prompt": "Architect a scalable microservices backend for an e-commerce platform using Docker, Spring Boot, and PostgreSQL. Explain how to handle database connections.",
        "expected_route": "cloud_llm"
    },
    
    # 4. Semantic Cache Hits (Expected: cache - close match to query 4)
    {
        "prompt": "Design a scalable backend of microservices for an online shopping site with Docker, Spring Boot, and PostgreSQL. Focus on DB connections.",
        "expected_route": "cache"
    },
    
    # 5. Simple queries
    {"prompt": "Quick response test.", "expected_route": "local_slm"}
]

async def check_health():
    """Verify gateway server is online before running benchmark."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{GATEWAY_URL}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"{GREEN}✔ Gateway Health Check Passed.{RESET}")
                print(f"  Local Ollama Connected: {data['local_ollama_connected']}")
                print(f"  Cloud Gemini Configured: {data['cloud_gemini_configured']}\n")
                return True
        except Exception:
            pass
    print(f"{RED}✖ Gateway is offline. Start the server using: uvicorn app.main:app --reload{RESET}")
    return False

async def query_gateway(prompt: str):
    """Send prompt to gateway and return JSON response."""
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(f"{GATEWAY_URL}/query", json={"prompt": prompt})
        response.raise_for_status()
        return response.json()

async def run_benchmark():
    """Runs the automated benchmark workload."""
    print(f"{BOLD}{BLUE}================================================================{RESET}")
    print(f"{BOLD}{BLUE}              LLM GATEWAY AUTOMATED PERFORMANCE BENCHMARK        {RESET}")
    print(f"{BOLD}{BLUE}================================================================{RESET}\n")

    if not await check_health():
        return

    # Clear cache before starting to ensure clean statistics
    async with httpx.AsyncClient() as client:
        await client.post(f"{GATEWAY_URL}/cache/clear")
        print(f"{YELLOW}Cleared semantic cache DB for clean benchmark run.{RESET}\n")

    total_queries = 0
    cache_hits = 0
    local_routes = 0
    cloud_routes = 0
    
    cache_latencies = []
    model_latencies = []
    
    total_cost_saved = 0.0
    estimated_gemini_cost_per_token_out = 0.0000003
    estimated_gemini_cost_per_token_in = 0.000000075
    
    print(f"{BOLD}{'#':<3} | {'Prompt Snippet':<45} | {'Route Source':<20} | {'Latency':<10} | {'Saved $':<8}{RESET}")
    print("-" * 95)

    for idx, item in enumerate(WORKLOAD, 1):
        prompt = item["prompt"]
        prompt_snippet = prompt[:42] + "..." if len(prompt) > 45 else prompt
        
        try:
            data = await query_gateway(prompt)
            latency = data["latency_ms"]
            source = data["source"]
            
            # Determine metric classification
            if "cache" in source:
                cache_hits += 1
                cache_latencies.append(latency)
                # Estimate cost saved by cache
                cost_saved = (len(prompt.split()) * 1.3 * estimated_gemini_cost_per_token_in) + (150 * estimated_gemini_cost_per_token_out)
                total_cost_saved += cost_saved
                cost_str = f"${cost_saved:.6f}"
                source_formatted = f"{GREEN}{source:<20}{RESET}"
            else:
                model_latencies.append(latency)
                cost_str = "$0.000000"
                if "local" in source:
                    local_routes += 1
                    source_formatted = f"{BLUE}{source:<20}{RESET}"
                    # Local runs save API costs compared to cloud
                    cost_saved = (len(prompt.split()) * 1.3 * estimated_gemini_cost_per_token_in) + (50 * estimated_gemini_cost_per_token_out)
                    total_cost_saved += cost_saved
                    cost_str = f"${cost_saved:.6f} (Local)"
                else:
                    cloud_routes += 1
                    source_formatted = f"{YELLOW}{source:<20}{RESET}"
            
            print(f"{idx:<3} | {prompt_snippet:<45} | {source_formatted} | {latency:>7.1f}ms | {cost_str}")
            total_queries += 1
            
        except Exception as e:
            print(f"{idx:<3} | {prompt_snippet:<45} | {RED}{'FAILED':<20}{RESET} | {'N/A':>9} | $0.000000")
            print(f"Error details: {e}")

    # Output Performance Dashboard
    avg_cache = sum(cache_latencies) / len(cache_latencies) if cache_latencies else 0.0
    avg_model = sum(model_latencies) / len(model_latencies) if model_latencies else 0.0
    speedup = avg_model / avg_cache if avg_cache > 0 else 0.0

    print(f"\n{BOLD}{BLUE}================================================================{RESET}")
    print(f"{BOLD}{BLUE}                   PERFORMANCE SUMMARY DASHBOARD                {RESET}")
    print(f"{BOLD}{BLUE}================================================================{RESET}")
    print(f"  Total Simulated Queries Run : {total_queries}")
    print(f"  Cache Hits (Semantic)       : {GREEN}{cache_hits}{RESET} / {total_queries} ({(cache_hits/total_queries)*100:.1f}%)")
    print(f"  Local SLM Routes            : {BLUE}{local_routes}{RESET}")
    print(f"  Cloud LLM Routes            : {YELLOW}{cloud_routes}{RESET}")
    print(f"  Average Cache Hit Latency   : {GREEN}{avg_cache:.2f} ms{RESET}")
    print(f"  Average Model Route Latency : {avg_model:.2f} ms")
    print(f"  Cache Acceleration Factor   : {BOLD}{GREEN}{speedup:.1f}x Faster{RESET}")
    print(f"  Total API Cost Saved (USD)  : {BOLD}{GREEN}${total_cost_saved:.6f}{RESET}")
    print(f"{BOLD}{BLUE}================================================================{RESET}\n")

async def run_interactive():
    """Starts an interactive REPL playground for the user/recruiter to test custom queries."""
    print(f"{BOLD}{GREEN}================================================================{RESET}")
    print(f"{BOLD}{GREEN}              DYNAMIC LLM ROUTER PLAYGROUND (INTERACTIVE)       {RESET}")
    print(f"{BOLD}{GREEN}================================================================{RESET}")
    print("Type any prompt you want. The gateway will dynamically:")
    print("  1. Check semantic cache (runs a real-time embedding similarity check).")
    print("  2. Classify prompt complexity (routes local vs. cloud dynamically).")
    print("  3. Run local SLM or cloud Gemini in real-time.")
    print("Type 'exit' or 'quit' to stop.\n")

    if not await check_health():
        return

    while True:
        try:
            prompt = input(f"{BOLD}User Prompt > {RESET}").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit"]:
                print("Exiting playground. Goodbye!")
                break
            
            print(f"{YELLOW}Processing query dynamically...{RESET}")
            start_time = time.time()
            data = await query_gateway(prompt)
            duration_ms = (time.time() - start_time) * 1000.0

            source = data["source"]
            response = data["response"]
            
            # Formatting outputs
            if "cache" in source:
                source_formatted = f"{GREEN}{source}{RESET}"
            elif "local" in source:
                source_formatted = f"{BLUE}{source}{RESET}"
            else:
                source_formatted = f"{YELLOW}{source}{RESET}"

            print(f"\n{BOLD}Gateway Routing Decision:{RESET}")
            print(f"  • Source Route : {source_formatted}")
            print(f"  • Latency      : {data['latency_ms']:.2f}ms (Total roundtrip: {duration_ms:.2f}ms)")
            print(f"  • Cost (USD)   : ${data['cost_usd']:.6f}")
            print(f"  • Response     : {BOLD}{response}{RESET}\n")
            print("-" * 64)

        except KeyboardInterrupt:
            print("\nExiting playground.")
            break
        except Exception as e:
            print(f"{RED}Error processing request: {e}{RESET}\n")

def main():
    parser = argparse.ArgumentParser(description="LLM Gateway Auditor and Benchmark Playground")
    parser.add_argument(
        "-i", "--interactive", 
        action="store_true", 
        help="Run in dynamic interactive mode to enter custom prompts."
    )
    args = parser.parse_args()

    if args.interactive:
        asyncio.run(run_interactive())
    else:
        asyncio.run(run_benchmark())

if __name__ == "__main__":
    main()
