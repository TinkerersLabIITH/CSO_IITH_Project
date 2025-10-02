from prometheus_client import start_http_server, Counter, Gauge
import time

        # Create a counter for total requests
requests_total = Counter('my_app_requests_total', 'Total number of requests to my application.')

        # Create a gauge for a specific value
current_value = Gauge('my_app_current_value', 'Current value of a specific metric.')

def process_request():
            requests_total.inc() # Increment the counter
            # Simulate some processing
            time.sleep(0.1)
            current_value.set(time.time() % 100) # Update the gauge

if __name__ == '__main__':
            # Start up the server to expose the metrics.
            start_http_server(8000)
            print("Prometheus metrics exposed on port 8000")
            while True:
                process_request()
                time.sleep(1)