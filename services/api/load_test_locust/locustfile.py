from locust import HttpUser, task, between, constant
import random

def generate_payload():
    payload = {}

    # V1 to V28 → float between 0 and 1
    for i in range(1, 29):
        payload[f"V{i}"] = random.uniform(0, 1)

    # Amount → float between 10 and 100
    payload["Amount"] = random.uniform(10, 100)

    return payload


class PredictUser(HttpUser):
    wait_time = constant(0)  # simulate user think time

    @task
    def predict(self):
        payload = generate_payload()

        self.client.post(
            "/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/predict"
        )