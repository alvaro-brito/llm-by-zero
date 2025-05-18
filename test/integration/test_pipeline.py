import unittest
import requests
import time
import json

class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:8000/api/v1"
        self.timeout = 300  # 5 minutes timeout for training
        
    def test_full_pipeline(self):
        # 1. Create model
        create_response = requests.post(f"{self.base_url}/models/create")
        self.assertEqual(create_response.status_code, 200)
        model_data = create_response.json()
        self.assertIn("model_id", model_data)
        model_id = model_data["model_id"]
        
        # 2. Wait for training completion
        start_time = time.time()
        while True:
            if time.time() - start_time > self.timeout:
                self.fail("Model training timeout")
            
            status_response = requests.get(f"{self.base_url}/models/{model_id}/status")
            self.assertEqual(status_response.status_code, 200)
            status = status_response.json()["status"]
            
            if status == "ready":
                break
            elif status == "failed":
                self.fail("Model training failed")
            
            time.sleep(10)
        
        # 3. Run evaluation
        eval_response = requests.post(f"{self.base_url}/models/{model_id}/evaluate")
        self.assertEqual(eval_response.status_code, 200)
        eval_results = eval_response.json()
        self.assertIn("perplexity", eval_results)
        self.assertIn("average_response_length", eval_results)
        
        # 4. Test inference
        test_input = {
            "text": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."
        }
        inference_response = requests.post(
            f"{self.base_url}/models/{model_id}/inference",
            json=test_input
        )
        self.assertEqual(inference_response.status_code, 200)
        inference_results = inference_response.json()
        self.assertIn("generated_text", inference_results)
        self.assertIn("confidence", inference_results)
        
        # 5. Verify response quality
        self.assertGreater(len(inference_results["generated_text"]), 0)
        self.assertGreater(inference_results["confidence"], 0.5)

if __name__ == '__main__':
    unittest.main() 