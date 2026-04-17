"""
Test Suite for Hybrid RAG System - Syllabus Query Validation
Tests the system's ability to provide accurate, cited responses from syllabus content
"""

import requests
import json
import time

class RagSystemTester:
    def __init__(self, base_url="http://127.0.0.1:5001"):
        self.base_url = base_url
        self.test_results = []
    
    def run_test_query(self, query, test_name):
        """Run a test query and return the response"""
        try:
            response = requests.post(
                self.base_url,
                data={'query': query},
                timeout=30
            )
            
            if response.status_code == 200:
                # Parse HTML response to extract answer
                html_content = response.text
                
                # Simple extraction - look for answer content
                # In a real test, you'd parse HTML properly
                return {
                    'success': True,
                    'query': query,
                    'test_name': test_name,
                    'response_html': html_content,
                    'status_code': response.status_code
                }
            else:
                return {
                    'success': False,
                    'query': query,
                    'test_name': test_name,
                    'error': f"HTTP {response.status_code}",
                    'response': response.text
                }
        
        except Exception as e:
            return {
                'success': False,
                'query': query,
                'test_name': test_name,
                'error': str(e)
            }
    
    def test_syllabus_textbooks(self):
        """Test 1: Query for syllabus textbooks - should return exact syllabus content"""
        query = "What are the textbooks to be referred for database management systems?"
        result = self.run_test_query(query, "Syllabus Textbooks Test")
        
        # Expected content check
        expected_authors = ['krishnan', 'gehrke', 'silberschatz', 'korth']
        expected_books = ['database management systems', 'database system concepts']
        
        if result['success']:
            response_lower = result['response_html'].lower()
            
            # Check for syllabus authors
            authors_found = [author for author in expected_authors if author in response_lower]
            books_found = [book for book in expected_books if book in response_lower]
            
            result['validation'] = {
                'authors_found': authors_found,
                'books_found': books_found,
                'has_citations': '[' in result['response_html'] and ']' in result['response_html'],
                'no_generic_content': 'comprehensive' not in response_lower and 'widely used' not in response_lower,
                'pass': len(authors_found) >= 2 and len(books_found) >= 1
            }
        
        return result
    
    def test_negative_case(self):
        """Test 2: Query for information not in syllabus - should refuse to answer"""
        query = "What are the latest NoSQL databases for big data?"
        result = self.run_test_query(query, "Negative Case Test")
        
        if result['success']:
            response_lower = result['response_html'].lower()
            
            # Should contain refusal message
            refusal_indicators = [
                'do not have enough information',
                'not found in the provided documents',
                'provided materials do not contain'
            ]
            
            result['validation'] = {
                'contains_refusal': any(indicator in response_lower for indicator in refusal_indicators),
                'no_generic_advice': 'mongodb' not in response_lower and 'cassandra' not in response_lower,
                'pass': any(indicator in response_lower for indicator in refusal_indicators)
            }
        
        return result
    
    def test_prerequisite_query(self):
        """Test 3: Query for prerequisites - should return specific syllabus content"""
        query = "What is the prerequisite subject for database management systems?"
        result = self.run_test_query(query, "Prerequisites Test")
        
        if result['success']:
            response_lower = result['response_html'].lower()
            
            # Should contain "data structures" from syllabus
            result['validation'] = {
                'mentions_data_structures': 'data structures' in response_lower,
                'has_citations': '[' in result['response_html'] and ']' in result['response_html'],
                'no_generic_list': response_lower.count('prerequisite') < 5,  # Not a long generic list
                'pass': 'data structures' in response_lower
            }
        
        return result
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🧪 Starting RAG System Test Suite...")
        print("=" * 50)
        
        tests = [
            self.test_syllabus_textbooks,
            self.test_negative_case,
            self.test_prerequisite_query
        ]
        
        results = []
        passed = 0
        
        for test_func in tests:
            print(f"\n⚡ Running {test_func.__name__}...")
            result = test_func()
            results.append(result)
            
            if result['success'] and result.get('validation', {}).get('pass', False):
                print(f"✅ PASSED: {result['test_name']}")
                passed += 1
            else:
                print(f"❌ FAILED: {result['test_name']}")
                if 'validation' in result:
                    print(f"   Details: {result['validation']}")
                else:
                    print(f"   Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
        
        return results

def main():
    """Run the test suite"""
    tester = RagSystemTester()
    
    # Wait a moment to ensure system is ready
    print("⏳ Waiting for system to be ready...")
    time.sleep(2)
    
    try:
        results = tester.run_all_tests()
        
        # Save detailed results
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to test_results.json")
        
        # Return success if all core tests pass
        core_passes = sum(1 for r in results if r.get('validation', {}).get('pass', False))
        return core_passes == len(results)
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)