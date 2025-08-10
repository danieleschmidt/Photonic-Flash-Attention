"""Internationalization and localization for photonic attention."""

import os
import json
import locale
from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from ..utils.logging import get_logger


logger = get_logger(__name__)


class PhotonicI18n:
    """Internationalization manager for photonic attention."""
    
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Español', 
        'fr': 'Français',
        'de': 'Deutsch',
        'ja': '日本語',
        'zh': '中文',
        'ko': '한국어',
        'pt': 'Português',
        'ru': 'Русский',
        'it': 'Italiano',
    }
    
    def __init__(self, language: Optional[str] = None, locale_dir: Optional[str] = None):
        """
        Initialize internationalization.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
            locale_dir: Directory containing translation files
        """
        self.locale_dir = Path(locale_dir) if locale_dir else self._get_default_locale_dir()
        self.current_language = language or self._detect_system_language()
        self.translations: Dict[str, Dict[str, str]] = {}
        self.fallback_language = 'en'
        
        # Load translations
        self._load_translations()
        
        logger.info(f"I18n initialized: language={self.current_language}")
    
    def _get_default_locale_dir(self) -> Path:
        """Get default locale directory."""
        return Path(__file__).parent / 'locales'
    
    def _detect_system_language(self) -> str:
        """Detect system language."""
        try:
            # Try environment variables first
            for env_var in ['LANG', 'LANGUAGE', 'LC_ALL', 'LC_MESSAGES']:
                if env_var in os.environ:
                    lang = os.environ[env_var].split('.')[0].split('_')[0].lower()
                    if lang in self.SUPPORTED_LANGUAGES:
                        return lang
            
            # Try Python locale
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                lang = system_locale.split('_')[0].lower()
                if lang in self.SUPPORTED_LANGUAGES:
                    return lang
        
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
        
        return 'en'  # Fallback to English
    
    def _load_translations(self) -> None:
        """Load translation files."""
        try:
            # Ensure locale directory exists
            self.locale_dir.mkdir(parents=True, exist_ok=True)
            
            # Load translations for each supported language
            for lang_code in self.SUPPORTED_LANGUAGES:
                translation_file = self.locale_dir / f"{lang_code}.json"
                
                if translation_file.exists():
                    try:
                        with open(translation_file, 'r', encoding='utf-8') as f:
                            self.translations[lang_code] = json.load(f)
                        logger.debug(f"Loaded {len(self.translations[lang_code])} translations for {lang_code}")
                    except Exception as e:
                        logger.error(f"Failed to load translations for {lang_code}: {e}")
                else:
                    # Create empty translation file
                    self.translations[lang_code] = {}
                    if lang_code == 'en':
                        self._create_default_english_translations()
                        self._save_translations(lang_code)
        
        except Exception as e:
            logger.error(f"Failed to load translations: {e}")
            self.translations = {'en': {}}
    
    def _create_default_english_translations(self) -> None:
        """Create default English translations."""
        self.translations['en'] = {
            # System messages
            'system.starting': 'Starting photonic attention system',
            'system.stopping': 'Stopping photonic attention system',
            'system.error': 'System error occurred',
            'system.ready': 'System ready',
            
            # Hardware messages
            'hardware.detecting': 'Detecting photonic hardware',
            'hardware.found': 'Found {count} photonic device(s)',
            'hardware.not_found': 'No photonic hardware detected',
            'hardware.error': 'Hardware error: {error}',
            'hardware.temperature_high': 'High temperature warning: {temp}°C',
            'hardware.power_high': 'High optical power warning: {power} mW',
            
            # Performance messages
            'performance.optimizing': 'Optimizing performance',
            'performance.benchmark': 'Running benchmark',
            'performance.result': 'Performance: {latency}ms latency, {throughput} ops/sec',
            
            # Security messages
            'security.validation_failed': 'Security validation failed',
            'security.access_denied': 'Access denied',
            'security.invalid_input': 'Invalid input detected',
            
            # Configuration messages
            'config.loading': 'Loading configuration',
            'config.invalid': 'Invalid configuration',
            'config.updated': 'Configuration updated',
            
            # Error messages
            'error.timeout': 'Operation timed out',
            'error.memory': 'Out of memory',
            'error.compute': 'Computation error',
            'error.network': 'Network error',
            
            # Status messages
            'status.initializing': 'Initializing',
            'status.running': 'Running',
            'status.stopped': 'Stopped',
            'status.error': 'Error',
            
            # Units and measurements
            'unit.milliseconds': 'ms',
            'unit.seconds': 's',
            'unit.celsius': '°C',
            'unit.milliwatts': 'mW',
            'unit.megabytes': 'MB',
            'unit.gigabytes': 'GB',
        }
    
    def _save_translations(self, language: str) -> None:
        """Save translations to file."""
        try:
            translation_file = self.locale_dir / f"{language}.json"
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(
                    self.translations.get(language, {}), 
                    f, 
                    indent=2, 
                    ensure_ascii=False,
                    sort_keys=True
                )
        except Exception as e:
            logger.error(f"Failed to save translations for {language}: {e}")
    
    def set_language(self, language: str) -> bool:
        """
        Set current language.
        
        Args:
            language: Language code
            
        Returns:
            True if language was set successfully
        """
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language: {language}")
            return False
        
        self.current_language = language
        logger.info(f"Language changed to: {language}")
        return True
    
    def get_available_languages(self) -> Dict[str, str]:
        """Get available languages."""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def translate(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """
        Translate message key.
        
        Args:
            key: Translation key (e.g., 'system.error')
            language: Optional language override
            **kwargs: Parameters for string formatting
            
        Returns:
            Translated message
        """
        target_language = language or self.current_language
        
        # Try target language first
        translations = self.translations.get(target_language, {})
        message = translations.get(key)
        
        # Fallback to English
        if not message and target_language != self.fallback_language:
            translations = self.translations.get(self.fallback_language, {})
            message = translations.get(key)
        
        # Final fallback to key itself
        if not message:
            message = key
            logger.warning(f"Missing translation: {key} for language {target_language}")
        
        # Format message with parameters
        try:
            if kwargs:
                return message.format(**kwargs)
            return message
        except Exception as e:
            logger.error(f"Translation formatting failed for '{key}': {e}")
            return message
    
    def t(self, key: str, **kwargs) -> str:
        """Shorthand for translate()."""
        return self.translate(key, **kwargs)
    
    def add_translation(self, language: str, key: str, value: str) -> None:
        """
        Add or update translation.
        
        Args:
            language: Language code
            key: Translation key
            value: Translation value
        """
        if language not in self.translations:
            self.translations[language] = {}
        
        self.translations[language][key] = value
        
        # Save to file
        self._save_translations(language)
        
        logger.debug(f"Added translation {language}.{key} = {value}")
    
    def get_locale_info(self) -> Dict[str, Any]:
        """Get current locale information."""
        try:
            loc = locale.getlocale()
            return {
                'language': self.current_language,
                'language_name': self.SUPPORTED_LANGUAGES.get(self.current_language, 'Unknown'),
                'system_locale': loc,
                'encoding': locale.getpreferredencoding(),
                'decimal_point': locale.localeconv().get('decimal_point', '.'),
                'thousands_sep': locale.localeconv().get('thousands_sep', ','),
                'loaded_translations': len(self.translations.get(self.current_language, {})),
                'fallback_language': self.fallback_language,
            }
        except Exception as e:
            logger.error(f"Failed to get locale info: {e}")
            return {
                'language': self.current_language,
                'error': str(e)
            }
    
    def format_number(self, number: float, decimals: int = 2) -> str:
        """Format number according to locale."""
        try:
            # Simple formatting - could be enhanced with locale-specific rules
            return f"{number:.{decimals}f}"
        except Exception:
            return str(number)
    
    def format_bytes(self, bytes_value: int, language: Optional[str] = None) -> str:
        """Format bytes with localized units."""
        units = [
            ('unit.bytes', 1),
            ('unit.kilobytes', 1024),
            ('unit.megabytes', 1024**2),
            ('unit.gigabytes', 1024**3),
            ('unit.terabytes', 1024**4),
        ]
        
        for unit_key, divisor in reversed(units):
            if bytes_value >= divisor:
                value = bytes_value / divisor
                unit = self.translate(unit_key, language)
                return f"{value:.1f} {unit}"
        
        return f"{bytes_value} {self.translate('unit.bytes', language)}"
    
    def create_translation_template(self, output_file: Optional[str] = None) -> Dict[str, str]:
        """
        Create translation template with all keys.
        
        Args:
            output_file: Optional file to save template
            
        Returns:
            Dictionary of all translation keys
        """
        all_keys = set()
        
        # Collect all keys from existing translations
        for translations in self.translations.values():
            all_keys.update(translations.keys())
        
        # Create template
        template = {key: f"TODO: Translate '{key}'" for key in sorted(all_keys)}
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(template, f, indent=2, ensure_ascii=False, sort_keys=True)
                logger.info(f"Translation template saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save translation template: {e}")
        
        return template


# Global I18n instance
_i18n_instance: Optional[PhotonicI18n] = None


def get_i18n() -> PhotonicI18n:
    """Get global I18n instance."""
    global _i18n_instance
    if _i18n_instance is None:
        _i18n_instance = PhotonicI18n()
    return _i18n_instance


def set_language(language: str) -> bool:
    """Set global language."""
    return get_i18n().set_language(language)


def translate(key: str, **kwargs) -> str:
    """Translate message (convenience function)."""
    return get_i18n().translate(key, **kwargs)


def t(key: str, **kwargs) -> str:
    """Shorthand for translate."""
    return translate(key, **kwargs)


def get_available_languages() -> Dict[str, str]:
    """Get available languages."""
    return get_i18n().get_available_languages()


def init_i18n(language: Optional[str] = None, locale_dir: Optional[str] = None) -> PhotonicI18n:
    """Initialize I18n system."""
    global _i18n_instance
    _i18n_instance = PhotonicI18n(language, locale_dir)
    return _i18n_instance