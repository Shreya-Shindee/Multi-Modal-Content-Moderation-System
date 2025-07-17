"""
System Status Report
===================

Final status report for the Multi-Modal Content Moderation System.
"""

import requests
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def check_api_health():
    """Check if the API is running and healthy."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def check_frontend():
    """Check if the frontend is accessible."""
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def test_api_endpoint():
    """Test a simple API endpoint."""
    try:
        response = requests.post(
            "http://localhost:8000/predict/text",
            json={"text": "This is a test message", "threshold": 0.5},
            timeout=10,
        )
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def main():
    """Generate system status report."""
    print("🛡️ Multi-Modal Content Moderation System - Status Report")
    print("=" * 65)

    # Check API Health
    print("🔍 Checking API Health...")
    api_healthy, health_data = check_api_health()
    if api_healthy:
        print("✅ API is healthy and running")
        print(f"   Status: {health_data.get('status', 'unknown')}")
        print(f"   Model loaded: {health_data.get('model_loaded', 'unknown')}")
        print(f"   Version: {health_data.get('version', 'unknown')}")
    else:
        print("❌ API is not responding")
        print(f"   Error: {health_data.get('error', 'Unknown error')}")

    # Check Frontend
    print("\n🖥️ Checking Frontend...")
    frontend_running = check_frontend()
    if frontend_running:
        print("✅ Frontend is running and accessible")
    else:
        print("❌ Frontend is not accessible")

    # Test API Functionality
    print("\n🧪 Testing API Functionality...")
    api_working, test_result = test_api_endpoint()
    if api_working:
        print("✅ API endpoint test successful")
        print(f"   Prediction: {test_result.get('prediction', 'unknown')}")
        print(f"   Confidence: {test_result.get('confidence', 0):.3f}")
        print(
            f"   Processing time: "
            f"{test_result.get('processing_time', 0):.3f}s"
        )
    else:
        print("❌ API endpoint test failed")
        print(f"   Error: {test_result.get('error', 'Unknown error')}")

    # System URLs
    print("\n🌐 System Access URLs:")
    print("   📊 API Documentation: http://localhost:8000/docs")
    print("   🖥️ Web Interface: http://localhost:8501")
    print("   ❤️ Health Check: http://localhost:8000/health")
    print("   📈 API Stats: http://localhost:8000/stats")

    # Component Status Summary
    print("\n📋 Component Status Summary:")
    print(f"   API Server: {'🟢 Running' if api_healthy else '🔴 Down'}")
    print(f"   Frontend: {'🟢 Running' if frontend_running else '🔴 Down'}")
    print(f"   End-to-End: {'🟢 Working' if api_working else '🔴 Failed'}")

    # Feature Capabilities
    print("\n🚀 Available Features:")
    print("   ✅ Text-only content analysis")
    print("   ✅ Image-only content analysis")
    print("   ✅ Multi-modal (text + image) analysis")
    print("   ✅ Batch text processing")
    print("   ✅ Real-time predictions with confidence scores")
    print("   ✅ Interactive web interface")
    print("   ✅ RESTful API with documentation")

    # Usage Examples
    print("\n💡 Quick Usage Examples:")
    print("   Text Analysis:")
    print("   curl -X POST http://localhost:8000/predict/text \\")
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"text": "Your text here", "threshold": 0.5}\'')
    print()
    print("   Web Interface:")
    print("   Open http://localhost:8501 in your browser")

    # Success Summary
    overall_success = api_healthy and frontend_running and api_working
    print("\n" + "=" * 65)
    if overall_success:
        print("🎉 SYSTEM FULLY OPERATIONAL!")
        print("   All components are running and functional.")
        print("   Ready for content moderation tasks!")
    else:
        print("⚠️ SYSTEM PARTIALLY OPERATIONAL")
        print("   Some components may need attention.")
        print("   Check individual component status above.")

    return overall_success


if __name__ == "__main__":
    # Wait a moment for services to fully start
    print("Waiting for services to initialize...")
    time.sleep(3)
    main()
