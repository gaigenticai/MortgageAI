"""
Configuration settings for MortgageAI agents.

This module centralizes all configuration for the AI agents,
database connections, and external service integrations.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Settings:
    """Application settings loaded from environment variables."""

    # Application Configuration
    NODE_ENV: str = os.getenv('NODE_ENV', 'development')
    REQUIRE_AUTH: bool = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'
    PORT: int = int(os.getenv('PORT', '3000'))
    HOST: str = os.getenv('HOST', '0.0.0.0')

    # Database Configuration
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://mortgage_user:mortgage_pass@localhost:5432/mortgage_db')
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PORT: int = int(os.getenv('DB_PORT', '5432'))
    DB_NAME: str = os.getenv('DB_NAME', 'mortgage_db')
    DB_USER: str = os.getenv('DB_USER', 'mortgage_user')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', 'mortgage_pass')
    DB_SSL: bool = os.getenv('DB_SSL', 'false').lower() == 'true'

    # Redis Configuration
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD')

    # AI Model Configuration
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    AI_MODEL: str = os.getenv('AI_MODEL', 'gpt-4')
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

    # OCR and Document Processing
    TESSERACT_DATA_PATH: str = os.getenv('TESSERACT_DATA_PATH', '/usr/share/tesseract-ocr/5/tessdata')
    OCR_LANGUAGES: str = os.getenv('OCR_LANGUAGES', 'eng+nld')

    # Authentication (when REQUIRE_AUTH=true)
    JWT_SECRET: str = os.getenv('JWT_SECRET', 'your_jwt_secret_here')
    JWT_EXPIRES_IN: str = os.getenv('JWT_EXPIRES_IN', '24h')
    BCRYPT_ROUNDS: int = int(os.getenv('BCRYPT_ROUNDS', '12'))

    # External APIs
    CREDIT_CHECK_API_URL: str = os.getenv('CREDIT_CHECK_API_URL', 'https://api.credit-check-service.com')
    KYC_API_URL: str = os.getenv('KYC_API_URL', 'https://api.kyc-service.com')
    LENDER_API_URL: str = os.getenv('LENDER_API_URL', 'https://api.lender-service.com')
    AFM_API_URL: str = os.getenv('AFM_API_URL', 'https://api.afm.nl')
    AFM_API_KEY: str = os.getenv('AFM_API_KEY', '')
    AFM_CACHE_TTL: int = int(os.getenv('AFM_CACHE_TTL', '3600'))  # 1 hour in seconds
    AFM_RATE_LIMIT_MAX: int = int(os.getenv('AFM_RATE_LIMIT_MAX', '100'))
    AFM_RATE_LIMIT_WINDOW: int = int(os.getenv('AFM_RATE_LIMIT_WINDOW', '3600'))  # 1 hour in seconds

    # File Upload Configuration
    UPLOAD_PATH: str = os.getenv('UPLOAD_PATH', '/app/uploads')
    MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE', '10485760'))  # 10MB
    ALLOWED_FILE_TYPES: list = os.getenv('ALLOWED_FILE_TYPES', 'pdf,jpg,jpeg,png').split(',')

    # Rate Limiting
    RATE_LIMIT_MAX: int = int(os.getenv('RATE_LIMIT_MAX', '100'))
    RATE_LIMIT_WINDOW_MS: int = int(os.getenv('RATE_LIMIT_WINDOW_MS', '900000'))  # 15 minutes

    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'info')
    LOG_FILE: str = os.getenv('LOG_FILE', '/app/logs/mortgage-ai.log')

    # Compliance Configuration
    REGULATION_UPDATE_INTERVAL: int = int(os.getenv('REGULATION_UPDATE_INTERVAL', '86400000'))  # 24 hours in ms
    READABILITY_TARGET_LEVEL: str = os.getenv('READABILITY_TARGET_LEVEL', 'B1')

    # Quality Control Configuration
    QC_COMPLETENESS_THRESHOLD: float = float(os.getenv('QC_COMPLETENESS_THRESHOLD', '95'))
    ANOMALY_DETECTION_THRESHOLD: float = float(os.getenv('ANOMALY_DETECTION_THRESHOLD', '0.8'))

    # Email Configuration
    SMTP_HOST: str = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    SMTP_PORT: int = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USER: str = os.getenv('SMTP_USER', '')
    SMTP_PASS: str = os.getenv('SMTP_PASS', '')
    FROM_EMAIL: str = os.getenv('FROM_EMAIL', 'noreply@mortgage-ai.com')

    def validate(self) -> bool:
        """
        Validate that all required settings are present.

        Returns:
            True if all required settings are valid
        """
        required_settings = [
            ('DATABASE_URL', self.DATABASE_URL),
        ]

        if 'claude' in self.AI_MODEL.lower():
            required_settings.append(('ANTHROPIC_API_KEY', self.ANTHROPIC_API_KEY))
        else:
            required_settings.append(('OPENAI_API_KEY', self.OPENAI_API_KEY))

        missing = []
        for name, value in required_settings:
            if not value or value == f'your_{name.lower()}_here':
                missing.append(name)

        if missing:
            logging.error(f"Missing required configuration: {', '.join(missing)}")
            return False

        return True

    def get_database_config(self) -> dict:
        """Get database configuration as dictionary."""
        return {
            'host': self.DB_HOST,
            'port': self.DB_PORT,
            'database': self.DB_NAME,
            'user': self.DB_USER,
            'password': self.DB_PASSWORD,
            'ssl': self.DB_SSL
        }

    def get_redis_config(self) -> dict:
        """Get Redis configuration as dictionary."""
        config = {
            'host': self.REDIS_HOST,
            'port': self.REDIS_PORT,
        }

        if self.REDIS_PASSWORD:
            config['password'] = self.REDIS_PASSWORD

        return config

# Global settings instance
settings = Settings()

# Validate settings on import
if not settings.validate():
    raise ValueError("Invalid configuration. Please check your environment variables.")
