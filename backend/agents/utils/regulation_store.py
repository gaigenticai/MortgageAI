"""
Regulation Store for AFM Compliance Rules

This module manages the storage and retrieval of AFM (Autoriteit Financiële Markten)
regulations and compliance rules for mortgage advice.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, ValidationError, field_validator

from ..config import settings
from ..database import get_db_connection


class RegulationData(BaseModel):
    """Pydantic model for regulation data validation."""
    regulation_code: str
    title: str
    content: str
    category: str
    effective_date: datetime
    expiry_date: Optional[datetime] = None
    keywords: List[str] = []
    source_url: Optional[str] = None
    last_updated: Optional[datetime] = None

    @field_validator('regulation_code')
    @classmethod
    def validate_regulation_code(cls, v):
        if not v or not v.strip():
            raise ValueError('regulation_code cannot be empty')
        return v.strip()

    @field_validator('effective_date', mode='before')
    @classmethod
    def parse_effective_date(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class AFMAPIClient:
    """Client for interacting with AFM APIs with caching, retry, and rate limiting."""

    def __init__(self):
        self.base_url = settings.AFM_API_URL.rstrip('/')
        self.api_key = settings.AFM_API_KEY
        self.cache = TTLCache(maxsize=100, ttl=settings.AFM_CACHE_TTL)
        self.rate_limit_max = settings.AFM_RATE_LIMIT_MAX
        self.rate_limit_window = settings.AFM_RATE_LIMIT_WINDOW
        self.request_count = 0
        self.window_start = datetime.utcnow()
        self.logger = logging.getLogger(__name__)

    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        now = datetime.utcnow()
        if (now - self.window_start).total_seconds() > self.rate_limit_window:
            self.request_count = 0
            self.window_start = now

        if self.request_count >= self.rate_limit_max:
            wait_time = self.rate_limit_window - (now - self.window_start).total_seconds()
            self.logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds")
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.window_start = datetime.utcnow()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated API request with retry logic."""
        await self._check_rate_limit()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}

        cache_key = f"{url}:{json.dumps(params, sort_keys=True)}"
        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key]

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, params=params, timeout=30) as response:
                    self.request_count += 1
                    response.raise_for_status()
                    data = await response.json()
                    self.cache[cache_key] = data
                    return data
            except aiohttp.ClientResponseError as e:
                if e.status == 429:  # Rate limited
                    self.logger.warning("AFM API rate limit hit, backing off")
                    raise
                self.logger.error(f"AFM API error: {e.status} - {e.message}")
                raise
            except Exception as e:
                self.logger.error(f"Error calling AFM API: {str(e)}")
                raise

    async def get_regulations(self, category: Optional[str] = None, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch regulations from AFM API."""
        params = {}
        if category:
            params['category'] = category
        if since:
            params['since'] = since.isoformat()

        try:
            data = await self._make_request('regulations', params)
            regulations = data.get('regulations', [])

            # Validate and transform data
            validated_regulations = []
            for reg in regulations:
                try:
                    validated = RegulationData(**reg)
                    validated_regulations.append(validated.model_dump())
                except ValidationError as e:
                    self.logger.warning(f"Invalid regulation data: {e}")
                    continue

            return validated_regulations
        except Exception as e:
            self.logger.error(f"Failed to fetch regulations: {str(e)}")
            return []

    async def get_regulation_updates(self, last_update: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch regulation updates since last update."""
        params = {}
        if last_update:
            params['since'] = last_update.isoformat()

        try:
            data = await self._make_request('regulations/updates', params)
            updates = data.get('updates', [])

            validated_updates = []
            for update in updates:
                try:
                    validated = RegulationData(**update)
                    validated_updates.append(validated.model_dump())
                except ValidationError as e:
                    self.logger.warning(f"Invalid update data: {e}")
                    continue

            return validated_updates
        except Exception as e:
            self.logger.error(f"Failed to fetch regulation updates: {str(e)}")
            return []


class RegulationStore:
    """
    Manages AFM regulation storage, updates, and compliance checking.

    Features:
    - Automated regulation ingestion from AFM APIs
    - Local storage with search capabilities
    - Compliance rule validation
    - Update monitoring and notifications
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.update_interval = timedelta(hours=24)  # Daily updates
        self.last_update = None
        self.api_client = AFMAPIClient()
        self.data_freshness_threshold = timedelta(hours=1)  # Consider data fresh for 1 hour

    async def initialize(self):
        """Initialize the regulation store and perform initial data load."""
        try:
            # Check if we need to update regulations
            await self._check_for_updates()

            # Ensure we have current regulations
            count = await self._get_regulation_count()
            if count == 0:
                await self._load_initial_regulations()

            self.logger.info(f"Regulation store initialized with {count} regulations")

        except Exception as e:
            self.logger.error(f"Failed to initialize regulation store: {str(e)}")
            raise

    async def get_relevant_regulations(self, context: str) -> List[Dict[str, Any]]:
        """
        Get regulations relevant to a specific context.

        Args:
            context: The context to find relevant regulations for (e.g., 'mortgage_advice')

        Returns:
            List of relevant regulation dictionaries
        """
        try:
            conn = await get_db_connection()

            # Simple keyword-based relevance (in production, use semantic search)
            query = """
            SELECT id, regulation_code, title, content, category,
                   effective_date, keywords, source_url
            FROM regulations
            WHERE is_active = true
              AND category = $1
            ORDER BY effective_date DESC
            """

            rows = await conn.fetch(query, context)
            regulations = []

            for row in rows:
                regulation = dict(row)
                # Parse keywords if stored as JSON
                if isinstance(regulation.get('keywords'), str):
                    try:
                        regulation['keywords'] = json.loads(regulation['keywords'])
                    except:
                        regulation['keywords'] = []
                regulations.append(regulation)

            await conn.close()
            return regulations

        except Exception as e:
            self.logger.error(f"Error getting relevant regulations: {str(e)}")
            return []

    async def search_regulations(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search regulations by text content.

        Args:
            query: Search query string
            category: Optional category filter

        Returns:
            List of matching regulations
        """
        try:
            conn = await get_db_connection()

            search_query = """
            SELECT id, regulation_code, title, content, category,
                   effective_date, source_url,
                   ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', $1)) as rank
            FROM regulations
            WHERE is_active = true
              AND to_tsvector('english', content) @@ plainto_tsquery('english', $1)
            """

            params = [query]

            if category:
                search_query += " AND category = $2"
                params.append(category)

            search_query += " ORDER BY rank DESC LIMIT 20"

            rows = await conn.fetch(search_query, *params)
            results = [dict(row) for row in rows]

            await conn.close()
            return results

        except Exception as e:
            self.logger.error(f"Error searching regulations: {str(e)}")
            return []

    async def add_regulation(self, regulation_data: Dict[str, Any]) -> bool:
        """
        Add a new regulation to the store.

        Args:
            regulation_data: Regulation data dictionary

        Returns:
            Success status
        """
        try:
            # Validate data using pydantic
            validated_data = RegulationData(**regulation_data)
            data = validated_data.model_dump()

            # Ensure last_updated is set
            data['last_updated'] = data.get('last_updated', datetime.utcnow())

            conn = await get_db_connection()

            # Check if regulation already exists
            existing = await conn.fetchrow(
                "SELECT id FROM regulations WHERE regulation_code = $1",
                data['regulation_code']
            )

            if existing:
                # Update existing regulation
                await conn.execute("""
                    UPDATE regulations SET
                        title = $2, content = $3, category = $4,
                        effective_date = $5, expiry_date = $6,
                        keywords = $7, source_url = $8, updated_at = CURRENT_TIMESTAMP
                    WHERE regulation_code = $1
                """,
                data['regulation_code'],
                data['title'],
                data['content'],
                data['category'],
                data['effective_date'],
                data.get('expiry_date'),
                json.dumps(data.get('keywords', [])),
                data.get('source_url')
                )
                self.logger.info(f"Updated regulation: {data['regulation_code']}")
            else:
                # Insert new regulation
                await conn.execute("""
                    INSERT INTO regulations (
                        regulation_code, title, content, category,
                        effective_date, expiry_date, keywords, source_url
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                data['regulation_code'],
                data['title'],
                data['content'],
                data['category'],
                data['effective_date'],
                data.get('expiry_date'),
                json.dumps(data.get('keywords', [])),
                data.get('source_url')
                )
                self.logger.info(f"Added new regulation: {data['regulation_code']}")

            await conn.close()
            return True

        except ValidationError as e:
            self.logger.error(f"Validation error for regulation data: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error adding regulation: {str(e)}")
            return False

    async def _check_for_updates(self):
        """Check if regulations need to be updated."""
        now = datetime.utcnow()

        if self.last_update is None or (now - self.last_update) > self.update_interval:
            self.logger.info("Checking for regulation updates...")
            await self._fetch_afm_updates()
            self.last_update = now

    def _is_data_fresh(self) -> bool:
        """Check if cached data is still fresh."""
        if self.last_update is None:
            return False
        return (datetime.utcnow() - self.last_update) < self.data_freshness_threshold

    async def _fetch_afm_updates(self):
        """Fetch updates from AFM API."""
        try:
            # Check data freshness
            if not self._is_data_fresh():
                self.logger.info("Data is stale, fetching full regulations")
                regulations = await self.api_client.get_regulations(category='mortgage_advice')
                for reg in regulations:
                    await self.add_regulation(reg)
            else:
                # Fetch only updates since last update
                updates = await self.api_client.get_regulation_updates(self.last_update)
                for update in updates:
                    await self.add_regulation(update)

        except Exception as e:
            self.logger.error(f"Error fetching AFM updates: {str(e)}")
            # Fallback to cached/mock data if API fails
            self.logger.info("Falling back to initial regulations")
            await self._load_initial_regulations()

    async def _load_initial_regulations(self):
        """Load initial set of AFM regulations."""
        try:
            # Try to fetch from API first
            regulations = await self.api_client.get_regulations(category='mortgage_advice')
            if regulations:
                for reg in regulations:
                    await self.add_regulation(reg)
                return
        except Exception as e:
            self.logger.warning(f"Failed to load from API, using fallback data: {str(e)}")

        # Fallback to static initial regulations
        initial_regs = self._get_initial_regulations()
        for reg in initial_regs:
            await self.add_regulation(reg)

    async def _get_regulation_count(self) -> int:
        """Get the total count of active regulations."""
        try:
            conn = await get_db_connection()
            result = await conn.fetchval("SELECT COUNT(*) FROM regulations WHERE is_active = true")
            await conn.close()
            return result or 0
        except Exception:
            return 0

    def _get_mock_afm_updates(self) -> List[Dict[str, Any]]:
        """Get mock AFM regulation updates for development."""
        return [
            {
                'regulation_code': 'WFT_ART_4_20',
                'title': 'Information Provision Requirements',
                'content': 'Financial service providers must provide clear, accurate, and not misleading information to consumers.',
                'category': 'mortgage_advice',
                'effective_date': datetime.utcnow().date(),
                'keywords': ['information', 'clear', 'accurate', 'misleading', 'transparency'],
                'source_url': 'https://www.afm.nl/en/professionals/regulation/wft'
            },
            {
                'regulation_code': 'WFT_ART_4_21',
                'title': 'Plain Language Requirements',
                'content': 'All communications must be in understandable language appropriate to the target audience.',
                'category': 'mortgage_advice',
                'effective_date': datetime.utcnow().date(),
                'keywords': ['plain language', 'understandable', 'communication', 'audience'],
                'source_url': 'https://www.afm.nl/en/professionals/regulation/wft'
            }
        ]

    def _get_initial_regulations(self) -> List[Dict[str, Any]]:
        """Get initial AFM regulations for mortgage advice."""
        return [
            {
                'regulation_code': 'BGFO_2019_1',
                'title': 'Mortgage Credit Directive Implementation',
                'content': 'Advisors must assess client creditworthiness and provide ESIS before contract conclusion.',
                'category': 'mortgage_advice',
                'effective_date': datetime(2019, 1, 1).date(),
                'keywords': ['creditworthiness', 'ESIS', 'assessment', 'contract'],
                'source_url': 'https://www.afm.nl/en/professionals/regulation/bgfo'
            },
            {
                'regulation_code': 'WFT_ART_34',
                'title': 'Advisory Agreement Requirements',
                'content': 'Advisors must have written agreements specifying scope of advice and remuneration.',
                'category': 'mortgage_advice',
                'effective_date': datetime(2018, 1, 1).date(),
                'keywords': ['advisory agreement', 'remuneration', 'scope', 'written'],
                'source_url': 'https://www.afm.nl/en/professionals/regulation/wft'
            },
            {
                'regulation_code': 'WFT_ART_35',
                'title': 'Suitability Assessment',
                'content': 'Advice must be suitable for client needs, objectives, and financial situation.',
                'category': 'mortgage_advice',
                'effective_date': datetime(2018, 1, 1).date(),
                'keywords': ['suitability', 'needs', 'objectives', 'financial situation'],
                'source_url': 'https://www.afm.nl/en/professionals/regulation/wft'
            }
        ]
