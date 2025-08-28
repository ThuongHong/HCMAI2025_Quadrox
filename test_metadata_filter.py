import requests
import json

# Test script for metadata filter feature
BASE_URL = "http://localhost:8000/api/v1/keyframe"

def test_metadata_filter():
    """Test the new metadata filter endpoint"""
    
    # Test 1: Basic search with author filter
    print("Test 1: Search with author filter")
    payload = {
        "query": "tin tức",
        "top_k": 5,
        "score_threshold": 0.0,  # Lower threshold
        "metadata_filter": {
            "authors": ["60 Giây Official"]
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search/metadata-filter", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Found {len(results.get('results', []))} results")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Search with keywords filter
    print("Test 2: Search with keywords filter")
    payload = {
        "query": "thời sự",
        "top_k": 5,
        "score_threshold": 0.0,  # Lower threshold
        "metadata_filter": {
            "keywords": ["tin tuc", "HTV"]  # These exist in the database
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search/metadata-filter", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Found {len(results.get('results', []))} results")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Search with length filter
    print("Test 3: Search with video length filter")
    payload = {
        "query": "news",
        "top_k": 5,
        "score_threshold": 0.0,  # Lower threshold
        "metadata_filter": {
            "min_length": 600,  # At least 10 minutes
            "max_length": 1800  # At most 30 minutes
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search/metadata-filter", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Found {len(results.get('results', []))} results")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 4: Search with title filter
    print("Test 4: Search with title contains filter")
    payload = {
        "query": "tin tức",
        "top_k": 5,
        "score_threshold": 0.0,  # Lower threshold
        "metadata_filter": {
            "title_contains": "60 Giây"
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search/metadata-filter", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Found {len(results.get('results', []))} results")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 5: Complex filter with multiple criteria
    print("Test 5: Complex filter with multiple criteria")
    payload = {
        "query": "tin tức mới nhất",
        "top_k": 10,
        "score_threshold": 0.0,  # Lower threshold
        "metadata_filter": {
            "authors": ["60 Giây"],
            "title_contains": "2024",
            "date_from": "01/08/2024"
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search/metadata-filter", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Found {len(results.get('results', []))} results")
            for i, result in enumerate(results.get('results', [])[:3]):
                print(f"  Result {i+1}: {result['path']} (score: {result['score']:.3f})")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")


    print("\n" + "="*50 + "\n")
    
    # Test 6: Date filter
    print("Test 6: Search with date filter")
    payload = {
        "query": "tin tức",
        "top_k": 5,
        "score_threshold": 0.0,
        "metadata_filter": {
            "date_from": "01/08/2024",
            "date_to": "31/08/2024"
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search/metadata-filter", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Found {len(results.get('results', []))} results")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    print("Testing Metadata Filter Feature")
    print("="*50)
    test_metadata_filter()
    print("\nTesting completed!")
