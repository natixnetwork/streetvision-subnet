#!/usr/bin/env python3
"""
Mock Organic Request Sender
===========================

This script sends mock organic requests to a running validator proxy to test
the organic task distribution system with real testnet miners.

Usage:
    python test_organic_requests.py --port 10913 --host localhost

Requirements:
    1. Start your validator with: python neurons/validator.py --proxy.port 10913
    2. Either use authentication bypass (see README) or provide valid credentials
"""

import asyncio
import base64
import json
import random
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np
from PIL import Image, ImageDraw


class OrganicRequestTester:
    """Sends mock organic requests to test the validator proxy."""
    
    def __init__(self, host: str = "localhost", port: int = 10913, 
                 use_auth_bypass: bool = True):
        self.base_url = f"http://{host}:{port}"
        self.use_auth_bypass = use_auth_bypass
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Test statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'duplicate_requests': 0,
            'rejected_requests': 0,
            'errors': []
        }
    
    def get_auth_header(self) -> Dict[str, str]:
        """Get authentication header for requests."""
        if self.use_auth_bypass:
            # Use test bypass (requires modification to validator_proxy.py)
            auth_token = base64.b64encode(b"test_bypass").decode()
        else:
            # TODO: Implement real authentication
            raise NotImplementedError("Real authentication not implemented yet")
        
        return {"Authorization": auth_token}
    
    def generate_test_image(self, width: int = 256, height: int = 256, 
                           image_type: str = "synthetic") -> str:
        """Generate a test image and return base64 encoded string."""
        
        if image_type == "synthetic":
            # Create synthetic image with random colors and shapes
            img = Image.new('RGB', (width, height), 
                          color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            draw = ImageDraw.Draw(img)
            
            # Add some random elements
            for _ in range(random.randint(3, 8)):
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2, y2 = random.randint(0, width), random.randint(0, height)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], 
                             fill=color)
        
        elif image_type == "noise":
            # Create noise image
            noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(noise)
        
        elif image_type == "gradient":
            # Create gradient image
            img = Image.new('RGB', (width, height))
            pixels = img.load()
            for x in range(width):
                for y in range(height):
                    pixels[x, y] = (int(255 * x / width), int(255 * y / height), 128)
        
        else:
            # Simple solid color
            img = Image.new('RGB', (width, height), color=(128, 128, 128))
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode()
    
    async def send_organic_request(self, image_b64: str, seed: Optional[int] = None) -> Dict:
        """Send a single organic request to the validator proxy."""
        
        if seed is None:
            seed = random.randint(0, int(1e9))
        
        payload = {
            "image": image_b64,
            "seed": seed
        }
        
        headers = {
            "Content-Type": "application/json",
            **self.get_auth_header()
        }
        
        try:
            self.stats['total_requests'] += 1
            
            print(f"🚀 Sending organic request (seed: {seed})...")
            response = await self.client.post(
                f"{self.base_url}/validator_proxy",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                self.stats['successful_requests'] += 1
                print(f"✅ Success! Task hash: {result.get('task_hash', 'N/A')}")
                print(f"   Miners queried: {result.get('miners_queried', 'N/A')}")
                print(f"   Valid responses: {result.get('valid_responses', 'N/A')}")
                print(f"   Predictions: {result.get('preds', 'N/A')}")
                return {'status': 'success', 'data': result}
            
            elif response.status_code == 429:
                self.stats['duplicate_requests'] += 1
                print(f"⚠️  Duplicate request detected")
                return {'status': 'duplicate', 'message': 'Duplicate task'}
            
            elif response.status_code == 503:
                self.stats['rejected_requests'] += 1
                print(f"❌ Request rejected: {response.text}")
                return {'status': 'rejected', 'message': response.text}
            
            else:
                self.stats['failed_requests'] += 1
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ Request failed: {error_msg}")
                self.stats['errors'].append(error_msg)
                return {'status': 'error', 'message': error_msg}
        
        except Exception as e:
            self.stats['failed_requests'] += 1
            error_msg = f"Exception: {str(e)}"
            print(f"❌ Request failed: {error_msg}")
            self.stats['errors'].append(error_msg)
            return {'status': 'error', 'message': error_msg}
    
    async def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        try:
            headers = self.get_auth_header()
            response = await self.client.get(
                f"{self.base_url}/health/liveness",
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_single_request(self):
        """Test a single organic request."""
        print("\n" + "="*60)
        print("SINGLE REQUEST TEST")
        print("="*60)
        
        # Test health first
        if not await self.test_health_check():
            return
        
        # Generate and send request
        image_b64 = self.generate_test_image(image_type="synthetic")
        result = await self.send_organic_request(image_b64)
        
        print(f"\nResult: {result}")
    
    async def test_duplicate_detection(self):
        """Test duplicate task detection."""
        print("\n" + "="*60)
        print("DUPLICATE DETECTION TEST")
        print("="*60)
        
        # Send same image twice
        image_b64 = self.generate_test_image(image_type="gradient")
        seed = 12345
        
        print("Sending first request...")
        result1 = await self.send_organic_request(image_b64, seed)
        
        print("\nSending duplicate request...")
        result2 = await self.send_organic_request(image_b64, seed)
        
        print(f"\nFirst result: {result1['status']}")
        print(f"Second result: {result2['status']}")
        
        if result1['status'] == 'success' and result2['status'] == 'duplicate':
            print("✅ Duplicate detection working correctly!")
        else:
            print("❌ Duplicate detection may not be working")
    
    async def test_concurrent_requests(self, num_requests: int = 5):
        """Test concurrent request handling."""
        print("\n" + "="*60)
        print(f"CONCURRENT REQUESTS TEST ({num_requests} requests)")
        print("="*60)
        
        # Generate different images for each request
        tasks = []
        for i in range(num_requests):
            image_b64 = self.generate_test_image(
                image_type=random.choice(["synthetic", "noise", "gradient"])
            )
            task = self.send_organic_request(image_b64, seed=1000 + i)
            tasks.append(task)
        
        # Send all requests concurrently
        print(f"Sending {num_requests} concurrent requests...")
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Analyze results
        successful = sum(1 for r in results if r['status'] == 'success')
        rejected = sum(1 for r in results if r['status'] == 'rejected')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        print(f"\n📊 Concurrent test results:")
        print(f"   Total time: {end_time - start_time:.2f}s")
        print(f"   Successful: {successful}")
        print(f"   Rejected: {rejected}")
        print(f"   Failed: {failed}")
    
    async def test_load_testing(self, num_requests: int = 20, delay: float = 1.0):
        """Test system under load."""
        print("\n" + "="*60)
        print(f"LOAD TEST ({num_requests} requests, {delay}s delay)")
        print("="*60)
        
        for i in range(num_requests):
            print(f"\n--- Request {i+1}/{num_requests} ---")
            
            image_b64 = self.generate_test_image(
                image_type=random.choice(["synthetic", "noise", "gradient"])
            )
            
            result = await self.send_organic_request(image_b64)
            
            if i < num_requests - 1:  # Don't delay after last request
                await asyncio.sleep(delay)
    
    def print_final_stats(self):
        """Print final test statistics."""
        print("\n" + "="*60)
        print("FINAL TEST STATISTICS")
        print("="*60)
        print(f"Total requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful_requests']}")
        print(f"Duplicates: {self.stats['duplicate_requests']}")
        print(f"Rejected: {self.stats['rejected_requests']}")
        print(f"Failed: {self.stats['failed_requests']}")
        
        if self.stats['total_requests'] > 0:
            success_rate = (self.stats['successful_requests'] / self.stats['total_requests']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        if self.stats['errors']:
            print(f"\nErrors encountered:")
            for error in self.stats['errors'][-5:]:  # Show last 5 errors
                print(f"  - {error}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test organic request system")
    parser.add_argument("--host", default="localhost", help="Validator proxy host")
    parser.add_argument("--port", type=int, default=10913, help="Validator proxy port")
    parser.add_argument("--test", choices=["single", "duplicate", "concurrent", "load", "all"], 
                       default="all", help="Test type to run")
    parser.add_argument("--num-requests", type=int, default=10, 
                       help="Number of requests for load/concurrent tests")
    parser.add_argument("--delay", type=float, default=2.0, 
                       help="Delay between requests for load test")
    parser.add_argument("--no-auth-bypass", action="store_true", 
                       help="Don't use authentication bypass")
    
    args = parser.parse_args()
    
    print("🧪 Mock Organic Request Tester")
    print("="*60)
    print(f"Target: {args.host}:{args.port}")
    print(f"Auth bypass: {'No' if args.no_auth_bypass else 'Yes'}")
    print("="*60)
    
    tester = OrganicRequestTester(
        host=args.host, 
        port=args.port, 
        use_auth_bypass=not args.no_auth_bypass
    )
    
    try:
        if args.test == "single" or args.test == "all":
            await tester.test_single_request()
        
        if args.test == "duplicate" or args.test == "all":
            await tester.test_duplicate_detection()
        
        if args.test == "concurrent" or args.test == "all":
            await tester.test_concurrent_requests(args.num_requests)
        
        if args.test == "load" or args.test == "all":
            await tester.test_load_testing(args.num_requests, args.delay)
    
    finally:
        tester.print_final_stats()
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())