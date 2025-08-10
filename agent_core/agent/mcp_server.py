from fastmcp import FastMCP
import httpx
import json
from typing import List, Optional, Dict, Any
import asyncio
import os
from bs4 import BeautifulSoup
import http.client
import urllib.parse
import re
import time
import html

mcp = FastMCP("Dynamic MCP Server", stateless_http=True)

@mcp.tool()
async def get_autogluon_tabular_workflow(specific_step: Optional[str] = None) -> str:
    """
    Extract AutoGluon TabularPredictor workflow from AWS Open Source Blog.
    Scrapes from: Machine learning with AutoGluon, an open source AutoML library | AWS Open Source Blog
    """
    result = {
        "source": "AWS Open Source Blog - AutoGluon Article",
        "timestamp": time.time(),
        "extracted_content": {},
        "error": None,
        "url_used": None
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Target the specific AWS Open Source Blog article about AutoGluon
            target_url = "https://aws.amazon.com/blogs/opensource/machine-learning-with-autogluon-an-open-source-automl-library/"
            
            try:
                response = await client.get(target_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    result["url_used"] = target_url
                    
                    # Remove navigation and UI elements
                    for element in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                        element.decompose()
                    
                    # Remove AWS-specific navigation elements
                    aws_nav_selectors = [
                        '.aws-header', '.aws-footer', '.aws-nav', '[class*="nav"]', 
                        '[class*="breadcrumb"]', '[class*="sidebar"]', '[class*="menu"]',
                        '[id*="nav"]', '.blog-sidebar', '.related-posts'
                    ]
                    for selector in aws_nav_selectors:
                        for element in soup.select(selector):
                            element.decompose()
                    
                    # Focus on the main blog post content
                    main_content = (soup.find('article') or 
                                  soup.find('.blog-post-content') or 
                                  soup.find('.post-content') or
                                  soup.find('main') or 
                                  soup.find('[role="main"]') or 
                                  soup.find('.content'))
                    
                    if main_content:
                        content_soup = main_content
                    else:
                        # Fallback to body content
                        content_soup = soup.find('body')
                    
                    # Extract meaningful paragraphs from the blog post
                    meaningful_paragraphs = []
                    for p in content_soup.find_all(['p', 'div'], class_=lambda x: not (x and any(nav_term in str(x).lower() for nav_term in ['nav', 'menu', 'sidebar', 'footer']))):
                        text = p.get_text(strip=True)
                        # Filter for AutoGluon-related content
                        if (len(text) > 60 and 
                            any(keyword in text.lower() for keyword in ['autogluon', 'tabular', 'predictor', 'machine learning', 'automl']) and
                            'toggle' not in text.lower() and 
                            'navigation' not in text.lower() and
                            'cookie' not in text.lower() and
                            not text.startswith('©')):
                            meaningful_paragraphs.append(text)
                    
                    # Extract workflow steps with AutoGluon-specific focus
                    workflow_steps = {
                        "initialization": [],
                        "data_preparation": [],
                        "fit_training": [],
                        "prediction": [], 
                        "evaluation": [],
                        "model_interpretation": [],
                        "deployment": []
                    }
                    
                    # AutoGluon-specific keyword mapping
                    keyword_mapping = {
                        "initialization": ["tabularpredictor", "import autogluon", "from autogluon", "predictor =", "ag."],
                        "data_preparation": ["load data", "dataset", "train_data", "test_data", "csv", "preprocess"],
                        "fit_training": ["fit(", "train", "training", ".fit", "hyperparameter", "time_limit"],
                        "prediction": ["predict(", "prediction", "inference", ".predict", "predict_proba"],
                        "evaluation": ["evaluate", "performance", "accuracy", "leaderboard", "score"],
                        "model_interpretation": ["feature importance", "interpret", "explain", "shap", "importance"],
                        "deployment": ["deploy", "save", "load", "production", "serving"]
                    }
                    
                    # Analyze content for workflow steps
                    all_content = meaningful_paragraphs
                    
                    for content in all_content:
                        content_lower = content.lower()
                        for step, keywords in keyword_mapping.items():
                            for keyword in keywords:
                                if keyword in content_lower:
                                    if content not in workflow_steps[step]:  # Avoid duplicates
                                        workflow_steps[step].append(content)
                    
                    # Store workflow steps in results (only non-empty ones)
                    for step, content_list in workflow_steps.items():
                        if content_list:
                            result["extracted_content"][step] = content_list[:3]  # Top 3 per step
                    
                    # Extract code examples specifically
                    code_blocks = content_soup.find_all(['code', 'pre', '.highlight'])
                    code_examples = []
                    for code in code_blocks:
                        code_text = code.get_text(strip=True)
                        if ('autogluon' in code_text.lower() or 
                            'TabularPredictor' in code_text or
                            'predictor' in code_text.lower()) and len(code_text) > 15:
                            code_examples.append(code_text)
                    
                    result["extracted_content"]["code_examples"] = code_examples[:5]  # Top 5 code examples
                    
                    # Extract any specific AutoGluon methods mentioned
                    method_patterns = [
                        r'TabularPredictor\.[a-zA-Z_]+\(',
                        r'predictor\.[a-zA-Z_]+\(',
                        r'\.fit\([^)]*\)',
                        r'\.predict\([^)]*\)',
                        r'\.evaluate\([^)]*\)'
                    ]
                    
                    methods_found = []
                    full_text = ' '.join(meaningful_paragraphs)
                    for pattern in method_patterns:
                        matches = re.findall(pattern, full_text)
                        methods_found.extend(matches)
                    
                    if methods_found:
                        result["extracted_content"]["autogluon_methods"] = list(set(methods_found))
                    
                    # Extract the article title and summary
                    title = soup.find('title')
                    if title:
                        result["extracted_content"]["article_title"] = title.get_text(strip=True)
                    
                    # Get first few paragraphs as summary
                    if meaningful_paragraphs:
                        result["extracted_content"]["article_summary"] = meaningful_paragraphs[:2]
                    
                    # Extract any performance metrics or benchmarks mentioned
                    benchmark_patterns = [
                        r'accuracy[:\s]+([0-9]+(?:\.[0-9]+)?%?)',
                        r'([0-9]+(?:\.[0-9]+)?%?)[:\s]*accuracy',
                        r'auc[:\s]+([0-9]+\.[0-9]+)',
                        r'time[:\s]+([0-9]+(?:\.[0-9]+)?(?:\s*(?:minutes?|seconds?|hours?)))',
                    ]
                    
                    benchmarks_found = {}
                    for pattern in benchmark_patterns:
                        matches = re.findall(pattern, full_text.lower())
                        if matches:
                            benchmarks_found[pattern] = matches[:3]
                    
                    if benchmarks_found:
                        result["extracted_content"]["performance_benchmarks"] = benchmarks_found
                
                else:
                    result["error"] = f"Failed to fetch AWS blog article. HTTP status: {response.status_code}"
                    
            except Exception as e:
                result["error"] = f"Error extracting from AWS blog: {str(e)}"
                
    except Exception as e:
        result["error"] = f"Error during extraction: {str(e)}"
    
    # Filter by specific step if requested
    if specific_step and specific_step in result["extracted_content"]:
        return json.dumps({
            "step": specific_step,
            "content": result["extracted_content"][specific_step],
            "source": result["source"],
            "url": result["url_used"]
        }, indent=2)
    
    return json.dumps(result, indent=2)



@mcp.tool()
async def search_and_extract(query: str, num_results: int = 5, content_type: str = "metrics", lr: str = "en-US") -> str:
    """
    Perform a web search using RapidAPI and extract clean content from the top result.
    
    Args:
        query: Search query string.
        num_results: Number of results to fetch.
        content_type: Type of content to extract ("metrics", "paragraphs", "numbers", "all").
        lr: Language/region code (default "en-US").
    """
    
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common unwanted patterns
        text = re.sub(r'^\s*[\|\-\•\*]+\s*', '', text)  # Remove bullet points at start
        text = re.sub(r'\s*\|\s*', ' | ', text)  # Normalize pipe separators
        
        return text.strip()
    
    def is_content_paragraph(element):
        """Check if an element contains meaningful content"""
        text = element.get_text(strip=True)
        
        # Skip if too short
        if len(text) < 30:
            return False
            
        # Skip navigation, ads, etc.
        skip_classes = ['nav', 'menu', 'footer', 'header', 'sidebar', 'ad', 'advertisement']
        element_classes = ' '.join(element.get('class', [])).lower()
        if any(skip_class in element_classes for skip_class in skip_classes):
            return False
            
        # Skip if mostly links or single sentences
        links = element.find_all('a')
        if len(links) > 3 and len(text) / len(links) < 20:
            return False
            
        return True
    
    result = {
        "query": query,
        "timestamp": time.time(),
        "top_link": None,
        "extracted_data": {},
        "error": None
    }
    
    try:
        # Load API key from dotenv
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            result["error"] = "Missing RAPIDAPI_KEY in environment"
            return json.dumps(result, indent=2)
        
        # Step 1: Perform search
        conn = http.client.HTTPSConnection("google-search72.p.rapidapi.com")
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "google-search72.p.rapidapi.com"
        }
        
        path = f"/search?q={urllib.parse.quote(query)}&num={num_results}&lr={lr}"
        conn.request("GET", path, headers=headers)
        res = conn.getresponse()
        
        if res.status != 200:
            result["error"] = f"Search failed with HTTP {res.status}: {res.reason}"
            return json.dumps(result, indent=2)
        
        parsed = json.loads(res.read().decode("utf-8"))
        top_result = parsed.get("results") or parsed.get("items") or []
        
        if not top_result or not top_result[0].get("link"):
            result["error"] = "No valid link found in search results"
            return json.dumps(result, indent=2)
        
        top_link = top_result[0]["link"]
        result["top_link"] = top_link
        
        # Step 2: Extract content with better filtering
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(top_link)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            extracted = {}
            
            # Extract paragraphs with better filtering
            if content_type in ["all", "paragraphs"]:
                paragraphs = []
                for p in soup.find_all(['p', 'div'], class_=lambda x: x != 'nav'):
                    if is_content_paragraph(p):
                        clean_text_content = clean_text(p.get_text())
                        if clean_text_content and len(clean_text_content) > 50:
                            paragraphs.append(clean_text_content)
                
                extracted["paragraphs"] = paragraphs[:5]  # Top 5 meaningful paragraphs
            
            # Extract numbers with better context
            if content_type in ["all", "numbers"]:
                main_content = soup.find('main') or soup.find('article') or soup.body
                if main_content:
                    text = clean_text(main_content.get_text())
                    
                    extracted["numbers"] = {
                        "percentages": list(set(re.findall(r'\b\d+(?:\.\d+)?%\b', text))),
                        "currencies": list(set(re.findall(r'[$£€¥]\d+(?:,\d{3})*(?:\.\d{2})?', text))),
                        "large_numbers": list(set(re.findall(r'\b\d{1,3}(?:,\d{3})+\b', text))),
                        "decimals": list(set(re.findall(r'\b\d+\.\d+\b', text)))
                    }
            
            # Extract metrics with cleaner context
            if content_type in ["all", "metrics"]:
                main_content = soup.find('main') or soup.find('article') or soup.body
                if main_content:
                    sentences = re.split(r'[.!?]+', main_content.get_text())
                    
                    metrics = {}
                    keywords = ['auc (area under the curve )', 'gini index in ml', 'precision in ml ', 'recall in ml ', 'f1 in ml', 'accuracy in ml', 'roc in ml', 'validation in ml ']
                    
                    for keyword in keywords:
                        matching_sentences = []
                        for sentence in sentences:
                            clean_sentence = clean_text(sentence)
                            if (keyword in clean_sentence.lower() and 
                                len(clean_sentence) > 20 and 
                                len(clean_sentence) < 200):
                                matching_sentences.append(clean_sentence)
                        
                        if matching_sentences:
                            metrics[keyword] = matching_sentences[:2]  # Top 2 relevant sentences
                    
                    extracted["metrics"] = metrics
            
            # Always extract clean title and headers
            title = soup.title.get_text(strip=True) if soup.title else "No title found"
            extracted["title"] = clean_text(title)
            
            headers = []
            for h in soup.find_all(['h1', 'h2', 'h3'])[:10]:  # Limit headers
                header_text = clean_text(h.get_text())
                if header_text and len(header_text) > 5:
                    headers.append(header_text)
            
            extracted["headers"] = headers
            result["extracted_data"] = extracted
    
    except Exception as e:
        result["error"] = f"Search and extract failed: {str(e)}"
    
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    mcp.run(transport="http")