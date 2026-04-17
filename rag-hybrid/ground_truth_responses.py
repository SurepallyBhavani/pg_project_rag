"""
Ground Truth Responses for RAG System Testing
These are the expected correct responses based on the actual syllabus content
"""

GROUND_TRUTH_RESPONSES = {
    "textbooks_query": {
        "query": "What are the textbooks to be referred for database management systems?",
        "expected_response": """Based on the provided syllabus materials:

**Textbooks:**
1. Database Management Systems by Raghurama Krishnan and Johannes Gehrke, Tata McGraw Hill 3rd Edition [syllabus_cse.pdf | Page 84]
2. Database System Concepts by Silberschatz and Korth, McGraw Hill V Edition [syllabus_cse.pdf | Page 84]

**References:**
1. Database Systems Design, Implementation, and Management by Peter Rob & Carlos Coronel, 7th Edition [syllabus_cse.pdf | Page 84]
2. SQL The Complete Reference by James R. Groff and Paul N. Weinberg, 3rd Edition [syllabus_cse.pdf | Page 84]
3. Oracle for Professionals by The X Team, S.Shah and V. Shah, SPD [syllabus_cse.pdf | Page 84]
4. Database Systems Using Oracle: A Simplified guide to SQL and PL/SQL by Shah, PHI [syllabus_cse.pdf | Page 84]

**Sources:**
- [syllabus_cse.pdf | Page 84] - Database Management Systems course textbook listing""",
        
        "validation_criteria": {
            "must_contain": [
                "Raghurama Krishnan",
                "Johannes Gehrke", 
                "Silberschatz",
                "Korth",
                "Tata McGraw Hill",
                "[syllabus_cse.pdf"
            ],
            "must_not_contain": [
                "comprehensive list",
                "widely used in academia",
                "Elmasri",
                "Navathe",
                "additional recommendations"
            ],
            "citation_format": "[filename | Page X]"
        }
    },
    
    "prerequisites_query": {
        "query": "What is the prerequisite subject for database management systems?",
        "expected_response": """Based on the provided syllabus materials:

The prerequisite for Database Management Systems is **"A course on Data Structures"** [syllabus_cse.pdf | Page 56].

**Sources:**
- [syllabus_cse.pdf | Page 56] - Prerequisites section stating "A course on Data Structures\"""",
        
        "validation_criteria": {
            "must_contain": [
                "Data Structures",
                "[syllabus_cse.pdf"
            ],
            "must_not_contain": [
                "typically depend on",
                "operating systems",
                "discrete mathematics",
                "comprehensive list"
            ]
        }
    },
    
    "negative_case": {
        "query": "What are the latest NoSQL databases for big data?",
        "expected_response": "Based on the provided materials, I do not have enough information to answer that question about NoSQL databases for big data. The available documents focus on relational database management systems and course syllabus content.",
        
        "validation_criteria": {
            "must_contain": [
                "do not have enough information" 
            ],
            "must_not_contain": [
                "MongoDB",
                "Cassandra", 
                "Redis",
                "comprehensive guide",
                "popular NoSQL"
            ]
        }
    }
}

def validate_response(query_type, actual_response):
    """
    Validate an actual response against ground truth criteria
    Returns validation results with pass/fail status
    """
    if query_type not in GROUND_TRUTH_RESPONSES:
        return {"error": f"Unknown query type: {query_type}"}
    
    criteria = GROUND_TRUTH_RESPONSES[query_type]["validation_criteria"]
    actual_lower = actual_response.lower()
    
    results = {
        "query_type": query_type,
        "must_contain_results": {},
        "must_not_contain_results": {},
        "citation_check": False,
        "overall_pass": False
    }
    
    # Check required content
    must_contain_pass = True
    for required in criteria["must_contain"]:
        found = required.lower() in actual_lower
        results["must_contain_results"][required] = found
        if not found:
            must_contain_pass = False
    
    # Check forbidden content  
    must_not_contain_pass = True
    for forbidden in criteria["must_not_contain"]:
        found = forbidden.lower() in actual_lower
        results["must_not_contain_results"][forbidden] = found
        if found:
            must_not_contain_pass = False
    
    # Check citations (if applicable)
    if "citation_format" in criteria:
        results["citation_check"] = "[" in actual_response and "]" in actual_response
    else:
        results["citation_check"] = True  # Not required
    
    # Overall pass/fail
    results["overall_pass"] = (
        must_contain_pass and 
        must_not_contain_pass and 
        results["citation_check"]
    )
    
    results["summary"] = {
        "must_contain_pass": must_contain_pass,
        "must_not_contain_pass": must_not_contain_pass,
        "citations_pass": results["citation_check"]
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    test_response = """Based on the provided syllabus materials:
    
    Textbooks:
    1. Database Management Systems by Raghurama Krishnan, Tata McGraw Hill
    [syllabus_cse.pdf | Page 84]"""
    
    validation = validate_response("textbooks_query", test_response)
    print("Validation Results:", validation)