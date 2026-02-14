"""
Test script to verify the entire pipeline works
Run this after training models to ensure everything is set up correctly
"""

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
from src.advisor import FestiveTravelAdvisor


def test_dataset_generation():
    """Test 1: Check if dataset can be generated"""
    print("\n" + "="*70)
    print("TEST 1: Dataset Generation")
    print("="*70)
    
    try:
        from src.generate_enhanced_dataset import generate_enhanced_dataset
        df = generate_enhanced_dataset(num_samples=100)
        
        assert len(df) == 100, "Dataset should have 100 rows"
        assert 'rush_level' in df.columns, "Dataset should have rush_level column"
        
        print("✅ Dataset generation successful!")
        print(f"   - Generated {len(df)} samples")
        print(f"   - Columns: {len(df.columns)}")
        return True
        
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        return False


def test_model_files_exist():
    """Test 2: Check if all model files exist after training"""
    print("\n" + "="*70)
    print("TEST 2: Model Files Existence")
    print("="*70)
    
    model_files = [
        "ml/models/rush_classifier.pkl",
        "ml/models/confirmation_regressor.pkl",
        "ml/models/booking_window_regressor.pkl",
        "ml/models/label_encoders.pkl",
        "ml/models/rush_target_encoder.pkl",
        "ml/models/rush_scaler.pkl",
        "ml/models/confirm_scaler.pkl",
        "ml/models/booking_scaler.pkl",
    ]
    
    all_exist = True
    for filepath in model_files:
        if os.path.exists(filepath):
            print(f"✅ Found: {filepath}")
        else:
            print(f"❌ Missing: {filepath}")
            all_exist = False
    
    if all_exist:
        print("\n✅ All model files exist!")
    else:
        print("\n❌ Some model files are missing. Run train_enhanced_models.py first!")
    
    return all_exist


def test_advisor_initialization():
    """Test 3: Check if advisor can be initialized"""
    print("\n" + "="*70)
    print("TEST 3: Advisor Initialization")
    print("="*70)
    
    try:
        advisor = FestiveTravelAdvisor()
        print("✅ Advisor initialized successfully!")
        return True, advisor
        
    except Exception as e:
        print(f"❌ Advisor initialization failed: {e}")
        return False, None


def test_rush_prediction(advisor):
    """Test 4: Test rush level prediction"""
    print("\n" + "="*70)
    print("TEST 4: Rush Level Prediction")
    print("="*70)
    
    try:
        result = advisor.predict_rush_level(
            festival="Diwali",
            days_before_festival=30,
            route_distance_km=1000,
            source_city_tier=1,
            destination_city_tier=2,
            train_class="Sleeper",
            train_type="Superfast"
        )
        
        print(f"✅ Rush prediction successful!")
        print(f"   - Rush Level: {result['rush_level']}")
        print(f"   - Confidence: {result['confidence']:.2%}")
        return True
        
    except Exception as e:
        print(f"❌ Rush prediction failed: {e}")
        return False


def test_confirmation_prediction(advisor):
    """Test 5: Test confirmation probability prediction"""
    print("\n" + "="*70)
    print("TEST 5: Confirmation Probability Prediction")
    print("="*70)
    
    try:
        prob = advisor.predict_confirmation_probability(
            current_waitlist_position=25,
            days_to_journey=15,
            train_type="Superfast",
            quota="General",
            train_class="Sleeper",
            historical_rush_index=75
        )
        
        print(f"✅ Confirmation prediction successful!")
        print(f"   - WL25 Confirmation Probability: {prob:.1%}")
        return True
        
    except Exception as e:
        print(f"❌ Confirmation prediction failed: {e}")
        return False


def test_booking_window_prediction(advisor):
    """Test 6: Test optimal booking window prediction"""
    print("\n" + "="*70)
    print("TEST 6: Optimal Booking Window Prediction")
    print("="*70)
    
    try:
        result = advisor.predict_optimal_booking_window(
            festival="Holi",
            route_distance_km=500,
            source_city_tier=1,
            destination_city_tier=2,
            train_class="3AC"
        )
        
        print(f"✅ Booking window prediction successful!")
        print(f"   - Optimal Window: {result['optimal_min']}-{result['optimal_max']} days before festival")
        return True
        
    except Exception as e:
        print(f"❌ Booking window prediction failed: {e}")
        return False


def test_complete_advisory(advisor):
    """Test 7: Test complete advisory system"""
    print("\n" + "="*70)
    print("TEST 7: Complete Advisory System")
    print("="*70)
    
    try:
        result = advisor.get_complete_advisory(
            festival="Diwali",
            days_before_festival=20,
            source_city="Delhi",
            destination_city="Patna",
            route_distance_km=1000,
            source_city_tier=1,
            destination_city_tier=2,
            train_class="Sleeper",
            train_type="Superfast",
            current_waitlist_position=40,
            quota="General"
        )
        
        print(f"✅ Complete advisory successful!")
        print(f"\n📊 Advisory Summary:")
        print(f"   - Rush Level: {result['rush_analysis']['rush_level']}")
        if result['confirmation_probability']:
            print(f"   - Confirmation Prob: {result['confirmation_probability']:.1%}")
        print(f"   - Recommendations: {len(result['recommendations'])} items found")
        return True
        
    except Exception as e:
        print(f"❌ Complete advisory failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🧪 "*30)
    print("FESTIVE TRAVEL ADVISOR - TEST SUITE")
    print("🧪 "*30)
    
    results = {}
    
    # Test 1: Dataset generation
    results['dataset'] = test_dataset_generation()
    
    # Test 2: Model files
    results['model_files'] = test_model_files_exist()
    
    if not results['model_files']:
        print("\n" + "⚠️ "*30)
        print("WARNING: Models not found! Please run:")
        print("  1. python src/generate_enhanced_dataset.py")
        print("  2. python src/train_enhanced_models.py")
        print("⚠️ "*30)
        return
    
    # Test 3: Advisor initialization
    success, advisor = test_advisor_initialization()
    results['advisor_init'] = success
    
    if not success or advisor is None:
        print("\n❌ Cannot proceed with further tests without advisor")
        return
    
    # Test 4-7: Individual predictions
    results['rush_pred'] = test_rush_prediction(advisor)
    results['confirm_pred'] = test_confirmation_prediction(advisor)
    results['booking_pred'] = test_booking_window_prediction(advisor)
    results['complete_advisory'] = test_complete_advisory(advisor)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Your system is ready to use!")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Please fix the issues above.")


if __name__ == "__main__":
    main()
