import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Force Python to read the .env file in your Codespace
load_dotenv() 

class SimulaAgent:
    def __init__(self):
        # Now it will successfully find GEMINI_API_KEY from the .env file
        self.client = genai.Client()
        self.system_prompt = """
        You are a deterministic data generator for Mechanistic Interpretability research.
        Generate Indirect Object Identification (IOI) text tasks.
        
        Rules:
        1. S1 and S2 must be distinct common English names.
        2. The sentence must end exactly before the indirect object token.
        3. Output ONLY a valid JSON array of objects. Do not include markdown formatting.
        
        Format:
        [
          {
            "text": "Mary and John went to the store. John handed a book to",
            "correct_target": " Mary",
            "incorrect_target": " John"
          }
        ]
        """

    def generate_batch(self, batch_size=5):
        """Generates a single batch of IOI prompts."""
        prompt = f"Generate exactly {batch_size} IOI JSON objects."
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                temperature=0.7
            )
        )
        return json.loads(response.text)

    def build_dataset(self, total_samples=10, batch_size=5, output_file="data/ioi_dataset.json"):
        """Generates the full dataset using staggered batches to respect API limits."""
        print(f"Initializing Simula Agent. Target: {total_samples} samples.")
        all_data = []
        batches = (total_samples + batch_size - 1) // batch_size

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        for i in range(batches):
            current_batch_size = min(batch_size, total_samples - len(all_data))
            print(f"Processing Batch {i+1}/{batches} ({current_batch_size} samples)...")
            
            try:
                batch_data = self.generate_batch(current_batch_size)
                all_data.extend(batch_data)
                print(f" -> Successfully retrieved {len(batch_data)} samples.")
            except Exception as e:
                print(f" -> Error on batch {i+1}: {e}")
                break

            if i < batches - 1:
                print(" -> Throttling for 4 seconds to respect API limits...")
                time.sleep(4)

        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\nDataset complete! Saved {len(all_data)} samples to {output_file}")
        return all_data

if __name__ == "__main__":
    agent = SimulaAgent()
    dataset = agent.build_dataset(total_samples=5, batch_size=5)