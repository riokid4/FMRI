import os
import json
import torch
from nnsight import LanguageModel
from circuit_tracer.attribution.attribute import attribute
from dotenv import load_dotenv

load_dotenv()

# --- THE PROXY WRAPPER ---
# This spoofs the ReplacementModel structure to trick circuit-tracer
class MockReplacementModel:
    def __init__(self, nnsight_model):
        self._nnsight_model = nnsight_model
        self.backend = "nnsight"
        self.tokenizer = nnsight_model.tokenizer
        self.cfg = getattr(nnsight_model._model, "config", None) 
        # Spoof an empty transcoder dictionary in case it tries to iterate over them
        self.transcoders = {} 

    def __getattr__(self, name):
        # Forward any other requested attributes to the real model
        return getattr(self._nnsight_model, name)
        
    def __call__(self, *args, **kwargs):
        return self._nnsight_model(*args, **kwargs)

class GemmaExtractor:
    def __init__(self, model_id="google/gemma-3-1b-it"):
        print(f"Loading {model_id} into memory...")
        
        if not torch.cuda.is_available():
            print("WARNING: No GPU detected! Running on CPU. This will be very slow.")
            self.device = "cpu"
        else:
            self.device = "cuda"
            print("GPU Detected.")

        raw_model = LanguageModel(
            model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            dispatch=True
        )
        
        # Wrap the raw model in our spoofed ReplacementModel
        self.model = MockReplacementModel(raw_model)
        print("Target AI Brain successfully loaded and spoofed!")

    def process_dataset(self, input_file="data/ioi_dataset.json", output_dir="eap_graphs"):
        os.makedirs(output_dir, exist_ok=True)

        with open(input_file, 'r') as f:
            dataset = json.load(f)

        print(f"Starting extraction for {len(dataset)} prompts...")

        for i, data in enumerate(dataset):
            text = data["text"]
            correct = data["correct_target"]

            print(f"\n[{i+1}/{len(dataset)}] Attributing: '{text[:50]}...'")

            try:
                graph = attribute(
                    prompt=text,
                    model=self.model,
                    attribution_targets=[correct]
                )

                if not hasattr(graph, 'edges') or not hasattr(graph, 'scores'):
                    print(f"Graph structure revealed: {dir(graph)}")
                    break

                edge_index = torch.tensor(graph.edges, dtype=torch.long)
                edge_scores = torch.tensor(graph.scores, dtype=torch.float32)
                n_nodes = graph.n_nodes

                sparse_tensor = torch.sparse_coo_tensor(
                    indices=edge_index,
                    values=edge_scores,
                    size=(n_nodes, n_nodes)
                ).coalesce()

                save_path = os.path.join(output_dir, f"graph_{i:04d}.pt")
                torch.save({
                    "sparse_graph": sparse_tensor,
                    "metadata": {
                        "text": text,
                        "target": correct,
                        "n_nodes": n_nodes,
                        "n_edges": edge_scores.shape[0]
                    }
                }, save_path)

                print(f"  -> Success! Extracted {edge_scores.shape[0]} active edges. Saved to {save_path}")

            except Exception as e:
                print(f"  -> FAILED on graph {i}: {e}")
                import traceback
                traceback.print_exc() 
                break

if __name__ == "__main__":
    extractor = GemmaExtractor()
    extractor.process_dataset()