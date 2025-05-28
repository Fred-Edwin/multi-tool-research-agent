"""Weather tool for fetching current weather information."""

import requests
import asyncio
from typing import Dict, Any, Optional
from config.settings import settings
from .base_tool import BaseTool, ToolInput, ToolOutput


class WeatherTool(BaseTool):
    """Tool for fetching weather information using OpenWeatherMap API."""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get current weather information for any location"
        )
        self.api_key = settings.weather_api_key
        self.base_url = settings.weather_base_url
    
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """Execute weather query for the given location."""
        try:
            query = input_data.query.strip()
            if not query:
                return self._create_error_output("Empty weather query provided")
            
            if not self.api_key:
                return self._create_error_output("Weather API key not configured")
            
            # Extract location from query
            location = self._extract_location(query)
            if not location:
                return self._create_error_output("Could not extract location from query")
            
            # Fetch weather data
            weather_data = await self._fetch_weather(location)
            if not weather_data:
                return ToolOutput(
                    result=f"Could not fetch weather data for {location}",
                    source="weather",
                    confidence=0.1,
                    metadata={"location": location}
                )
            
            # Format weather information
            formatted_result = self._format_weather_data(weather_data, location)
            
            return ToolOutput(
                result=formatted_result,
                source="weather",
                confidence=0.9,
                metadata={
                    "location": location,
                    "temperature": weather_data.get("main", {}).get("temp"),
                    "condition": weather_data.get("weather", [{}])[0].get("description")
                }
            )
            
        except Exception as e:
            self.logger.error(f"Weather tool error: {str(e)}")
            return self._create_error_output(f"Weather query failed: {str(e)}")
    
    def _extract_location(self, query: str) -> str:
        """Extract location from weather query."""
        # Remove common weather-related words to isolate location
        weather_words = [
            "weather", "temperature", "temp", "climate", "forecast",
            "what's", "what is", "how's", "how is", "in", "at", "for",
            "the", "current", "today", "now", "like"
        ]
        
        words = query.lower().split()
        location_words = []
        
        for word in words:
            # Remove punctuation
            clean_word = word.strip('.,!?')
            if clean_word not in weather_words and clean_word:
                location_words.append(clean_word)
        
        # Join remaining words as location
        location = ' '.join(location_words).strip()
        
        # If we couldn't extract location, try alternative patterns
        if not location:
            # Look for patterns like "weather in Tokyo" or "Tokyo weather"
            import re
            patterns = [
                r'weather\s+in\s+([^?]+)',
                r'weather\s+for\s+([^?]+)',
                r'([^?]+)\s+weather',
                r'temperature\s+in\s+([^?]+)',
                r'([^?]+)\s+temperature'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    location = match.group(1).strip()
                    break
        
        return location
    
    async def _fetch_weather(self, location: str) -> Optional[Dict[str, Any]]:
        """Fetch weather data from OpenWeatherMap API."""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'  # Use Celsius
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Weather API request failed: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Weather data fetch error: {str(e)}")
            return None
    
    def _format_weather_data(self, data: Dict[str, Any], location: str) -> str:
        """Format weather data into readable text."""
        try:
            # Extract key information
            main = data.get('main', {})
            weather = data.get('weather', [{}])[0]
            wind = data.get('wind', {})
            
            temp = main.get('temp', 'N/A')
            feels_like = main.get('feels_like', 'N/A')
            humidity = main.get('humidity', 'N/A')
            description = weather.get('description', 'N/A').title()
            wind_speed = wind.get('speed', 'N/A')
            
            # Convert temperature to Fahrenheit for additional info
            temp_f = (temp * 9/5) + 32 if isinstance(temp, (int, float)) else 'N/A'
            
            result = f"Current weather in {location.title()}:\n\n"
            result += f"Temperature: {temp}°C ({temp_f:.1f}°F)\n" if isinstance(temp_f, float) else f"Temperature: {temp}°C\n"
            result += f"Feels like: {feels_like}°C\n"
            result += f"Condition: {description}\n"
            result += f"Humidity: {humidity}%\n"
            result += f"Wind speed: {wind_speed} m/s\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Weather formatting error: {str(e)}")
            return f"Weather data available for {location}, but formatting failed."
    
    def is_relevant(self, query: str) -> bool:
        """Check if weather tool is relevant for the query."""
        weather_keywords = [
            "weather", "temperature", "temp", "climate", "forecast",
            "hot", "cold", "rain", "snow", "sunny", "cloudy",
            "humidity", "wind", "storm"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in weather_keywords)