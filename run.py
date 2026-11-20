"""
run.py: Standalone Interactive Runner for the Native Legal Transformer LLM.

Run in your terminal or inside VS Code:
    python3 run.py
"""

import sys
import time
from pathlib import Path

from engine import LegalLLMEngine
from trainer import train_legal_llm


def banner():
    print("=" * 80)
    print("⚖️  LEGAL TRANSFORMER LLM — FROM-SCRATCH DECODER ARCHITECTURE")
    print("    100% Offline | Zero Cloud APIs | Native PyTorch on Apple Silicon GPU")
    print("=" * 80)


def show_summary():
    engine = LegalLLMEngine.get_instance()
    info = engine.get_summary()
    print("\n📊 MODEL ARCHITECTURE SPECIFICATIONS:")
    print("-" * 60)
    print(f"  • Architecture:           Decoder-Only Autoregressive Transformer")
    print(f"  • Total Parameters:       {info['total_params']:,} ({info['total_params'] / 1e6:.2f}M)")
    print(f"  • Transformer Layers:     {info['layers']} Blocks (Pre-LayerNorm)")
    print(f"  • Multi-Head Attention:   {info['heads']} Heads (d_k = {info['d_model'] // info['heads']})")
    print(f"  • Hidden Dimension:       d_model = {info['d_model']}")
    print(f"  • Feed-Forward Dimension: d_ff = {info['d_model'] * 4} (GeLU Activation)")
    print(f"  • Max Sequence Length:    {info['max_seq_len']} Tokens")
    print(f"  • Vocabulary Size:        {info['vocab_size']:,} Legal Subword Tokens")
    print(f"  • Compute Device:         {info['device_name']} ({info['device']})")
    print(f"  • Checkpoint Status:      {'Loaded from disk' if info['is_checkpoint_loaded'] else 'Clean Initialized'}")
    print("-" * 60)


def interactive_generate():
    engine = LegalLLMEngine.get_instance()
    print("\n[PROMPT GENERATION]")
    print("Enter any legal prompt or question:")
    print("Example 1: Can my employer enforce a 2-year non-compete bond after I resign from my software job in India?")
    print("Example 2: My parents are threatening to disown me because I want to marry someone they don't approve of. What are my legal rights as an adult?")
    print("Example 3: My friend shared my private WhatsApp chats with other people without my permission. Can I take legal action against them in India?")
    print("Example 4: What is Article 21 of the Indian Constitution and what rights does it protect?")
    prompt = input("\nPrompt > ").strip(' \t\n\r"\'')
    if not prompt:
        prompt = "Can my employer enforce a 2-year non-compete bond after I resign from my software job in India?"
        print(f"Using default: \"{prompt}\"")

    print(f"\n⚡ Generating on {engine.device_name}...\n")
    print("Output: \"", end="", flush=True)

    token_count = 0
    for word in engine.stream_generate(prompt, max_tokens=75, temperature=0.2, top_k=5):
        print(word, end=" ", flush=True)
        time.sleep(0.015)
        token_count += 1

    print(f"\"\n\n[Synthesized {token_count} neural tokens | 100% Local]")


def test_clause_risk():
    engine = LegalLLMEngine.get_instance()
    print("\n[CLAUSE RISK ANALYSIS DEMO]")
    clause = "Vendor shall defend, indemnify, and hold harmless Client, its officers, and affiliates against any and all losses, liabilities, claims, indirect damages, and lost profits without limitation."
    print(f"Target Clause: \"{clause}\"")
    prompt = f"<bos> <clause> {clause} <risk>"
    print(f"\n⚡ Synthesizing risk explanation on {engine.device_name}...")
    print("Risk Finding: \"", end="", flush=True)

    for word in engine.stream_generate(prompt, max_tokens=65, temperature=0.2, top_k=5):
        print(word, end=" ", flush=True)
        time.sleep(0.015)
    print("\"\n")


def test_clause_rewrite():
    engine = LegalLLMEngine.get_instance()
    print("\n[CLAUSE REWRITING DEMO]")
    clause = "All disputes arising under this Agreement shall be referred to a sole arbitrator appointed exclusively by the Managing Director of the Company alone."
    print(f"Aggressive Clause: \"{clause}\"")
    prompt = f"<bos> <clause> {clause} <rewrite>"
    print(f"\n⚡ Synthesizing balanced revision on {engine.device_name}...")
    print("Proposed Revision: \"", end="", flush=True)

    for word in engine.stream_generate(prompt, max_tokens=65, temperature=0.2, top_k=5):
        print(word, end=" ", flush=True)
        time.sleep(0.015)
    print("\"\n")


def run_training():
    print("\n[RETRAIN TRANSFORMER WEIGHTS]")
    epochs_input = input("Enter number of training epochs (default: 4): ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else 4
    train_legal_llm(epochs=epochs, batch_size=16)


def main():
    banner()

    # CLI flag shortcuts
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--summary", "-s"]:
            show_summary()
            return
        elif arg in ["--train", "-t"]:
            train_legal_llm(epochs=4, batch_size=16)
            return
        elif arg in ["--demo", "-d"]:
            show_summary()
            test_clause_risk()
            test_clause_rewrite()
            return
        elif arg in ["--prompt", "-p"] and len(sys.argv) > 2:
            prompt = " ".join(sys.argv[2:])
            engine = LegalLLMEngine.get_instance()
            print(f"Generating for prompt: \"{prompt}\"")
            print("Output: ", end="")
            for w in engine.stream_generate(f"<bos> <user> {prompt} <sep> <assistant>", max_tokens=50):
                print(w, end=" ", flush=True)
            print()
            return

    while True:
        print("\nChoose an option (or type your question directly):")
        print("  1. Ask a Question / Interactive Prompt (Type 1, then enter question)")
        print("  2. Test Clause Risk Analysis Demo (<clause> -> <risk>)")
        print("  3. Test Clause Rewriting Demo (<clause> -> <rewrite>)")
        print("  4. Inspect Model Architecture & Parameters")
        print("  5. Retrain Model Weights on GPU")
        print("  6. Exit")

        try:
            choice = input("\nEnter choice (1-6) or ask a question directly > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Legal Transformer LLM. Goodbye!")
            break

        if not choice:
            continue
        elif choice == "1":
            interactive_generate()
        elif choice == "2":
            test_clause_risk()
        elif choice == "3":
            test_clause_rewrite()
        elif choice == "4":
            show_summary()
        elif choice == "5":
            run_training()
        elif choice in ["6", "exit", "quit", "q"]:
            print("\nExiting Legal Transformer LLM. Goodbye!")
            break
        elif len(choice) > 3 or any(w in choice.lower() for w in ["what", "how", "can", "is", "article", "section", "clause", "employee", "vendor", "chat", "contract", "legal"]):
            # User directly typed a question instead of typing 1!
            engine = LegalLLMEngine.get_instance()
            print(f"\n⚡ Generating answer on {engine.device_name}...\n")
            print("Output: \"", end="", flush=True)
            token_count = 0
            formatted = f"<bos> <user> {choice} <sep> <assistant>"
            for word in engine.stream_generate(formatted, max_tokens=75, temperature=0.2, top_k=5):
                print(word, end=" ", flush=True)
                time.sleep(0.015)
                token_count += 1
            print(f"\"\n\n[Synthesized {token_count} neural tokens | 100% Local]")
        else:
            print("Invalid selection. Please choose a number from 1 to 6, or type your question directly.")


if __name__ == "__main__":
    main()
