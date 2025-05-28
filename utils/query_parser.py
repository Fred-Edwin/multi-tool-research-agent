"""Query parsing utilities for the Research Agent."""

import re
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from config.settings import settings


class QueryParser:
    """Parses complex queries into sub-queries and determines tool requirements."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.3,
            max_tokens=500
        )
    
    async def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a complex query into actionable sub-queries.
        
        Args:
            query: The original user query
            
        Returns:
            Dict containing parsed information
        """
        try:
            # Use LLM to break down the query
            breakdown = await self._llm_parse_query(query)
            
            # Extract sub-queries and required tools
            sub_queries = self._extract_sub_queries(breakdown)
            required_tools = self._determine_required_tools(query, sub_queries)
            
            return {
                'original_query': query,
                'sub_queries': sub_queries,
                'required_tools': required_tools,
                'query_type': self._classify_query_type(query),
                'complexity': self._assess_complexity(query, sub_queries)
            }
            
        except Exception as e:
            # Fallback to simple parsing if LLM fails
            return self._simple_parse(query)
    
    async def _llm_parse_query(self, query: str) -> str:
        """Use LLM to break down complex queries."""
        system_prompt = """You are a query analysis assistant. Break down complex queries into simple, actionable sub-queries that can be handled by different tools.

Available tools:
- web_search: For current information and news
- calculator: For mathematical calculations
- weather: For weather information
- wikipedia: For factual/encyclopedic information

Instructions:
1. Identify if the query needs multiple tools
2. Break complex queries into 2-5 simple sub-queries
3. Each sub-query should be specific and actionable
4. Maintain logical order for dependent queries

Format your response as:
SUB_QUERIES:
1. [sub-query 1]
2. [sub-query 2]
...

TOOLS_NEEDED:
- tool_name: reason why needed
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def _extract_sub_queries(self, llm_response: str) -> List[str]:
        """Extract sub-queries from LLM response."""
        sub_queries = []
        
        # Look for numbered list in the response
        lines = llm_response.split('\n')
        in_sub_queries_section = False
        
        for line in lines:
            line = line.strip()
            if 'SUB_QUERIES:' in line or 'SUB-QUERIES:' in line:
                in_sub_queries_section = True
                continue
            elif 'TOOLS_NEEDED:' in line or line.startswith('TOOLS'):
                in_sub_queries_section = False
                continue
            
            if in_sub_queries_section and line:
                # Extract numbered items
                match = re.match(r'^\d+\.\s*(.+)', line)
                if match:
                    sub_queries.append(match.group(1).strip())
                elif line and not line.startswith('-'):
                    # Sometimes the LLM doesn't number items
                    sub_queries.append(line)
        
        return sub_queries[:5]  # Limit to 5 sub-queries
    
    def _determine_required_tools(self, query: str, sub_queries: List[str]) -> List[str]:
        """Determine which tools are needed based on query analysis."""
        tools_needed = set()
        all_queries = [query] + sub_queries
        
        for q in all_queries:
            q_lower = q.lower()
            
            # Check for calculator needs
            if any(keyword in q_lower for keyword in ['calculate', 'math', 'compute', '%', 'interest']) or re.search(r'\d+.*[\+\-\*/].*\d+', q):
                tools_needed.add('calculator')
            
            # Check for weather needs
            if any(keyword in q_lower for keyword in ['weather', 'temperature', 'climate']):
                tools_needed.add('weather')
            
            # Check for Wikipedia needs
            if any(keyword in q_lower for keyword in ['who is', 'what is', 'history', 'biography', 'definition']):
                tools_needed.add('wikipedia')
            
            # Check for web search needs
            if any(keyword in q_lower for keyword in ['current', 'recent', 'news', 'latest', 'today']):
                tools_needed.add('web_search')
        
        return list(tools_needed)
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['calculate', 'compute', 'math']):
            return 'calculation'
        elif any(word in query_lower for word in ['weather', 'temperature']):
            return 'weather'
        elif any(word in query_lower for word in ['who is', 'what is', 'biography']):
            return 'factual'
        elif any(word in query_lower for word in ['current', 'recent', 'news']):
            return 'current_info'
        elif 'compare' in query_lower or 'vs' in query_lower:
            return 'comparison'
        else:
            return 'general'
    
    def _assess_complexity(self, query: str, sub_queries: List[str]) -> str:
        """Assess the complexity of the query."""
        # Count complexity indicators
        complexity_score = 0
        
        # Multiple sub-queries indicate complexity
        complexity_score += len(sub_queries)
        
        # Multiple question words
        question_words = ['who', 'what', 'when', 'where', 'why', 'how']
        complexity_score += sum(1 for word in question_words if word in query.lower())
        
        # Comparison or analysis keywords
        complex_keywords = ['compare', 'analyze', 'vs', 'versus', 'difference', 'relationship']
        complexity_score += sum(1 for keyword in complex_keywords if keyword in query.lower())
        
        # Length of query
        if len(query.split()) > 15:
            complexity_score += 1
        
        if complexity_score >= 4:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _simple_parse(self, query: str) -> Dict[str, Any]:
        """Fallback simple parsing if LLM fails."""
        return {
            'original_query': query,
            'sub_queries': [query],  # Treat as single query
            'required_tools': self._determine_required_tools(query, []),
            'query_type': self._classify_query_type(query),
            'complexity': 'low'
        }