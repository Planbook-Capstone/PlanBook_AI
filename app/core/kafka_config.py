"""
Kafka Configuration for PlanBook AI
Cấu hình Kafka để kết nối với SpringBoot
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class KafkaSettings(BaseSettings):
    """Kafka configuration settings"""
    
    # Kafka Bootstrap Servers
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    
    # Topic Configuration
    KAFKA_TOPIC_NAME: str = os.getenv("KAFKA_TOPIC_NAME", "planbook")
    
    # Producer Configuration
    KAFKA_PRODUCER_CONFIG: dict = {
        "bootstrap_servers": None,  # Will be set from KAFKA_BOOTSTRAP_SERVERS
        "value_serializer": lambda v: v.encode('utf-8') if isinstance(v, str) else v,
        "key_serializer": lambda k: k.encode('utf-8') if isinstance(k, str) else k,
        "acks": "all",  # Wait for all replicas to acknowledge
        "retries": 3,
        "retry_backoff_ms": 100,
        "request_timeout_ms": 30000,
        "delivery_timeout_ms": 120000,
    }
    
    # Consumer Configuration
    KAFKA_CONSUMER_CONFIG: dict = {
        "bootstrap_servers": None,  # Will be set from KAFKA_BOOTSTRAP_SERVERS
        "value_deserializer": lambda m: m.decode('utf-8') if m else None,
        "key_deserializer": lambda m: m.decode('utf-8') if m else None,
        "group_id": "planbook-fastapi-group",
        "auto_offset_reset": "latest",  # Start from latest messages
        "enable_auto_commit": True,
        "auto_commit_interval_ms": 1000,
        "session_timeout_ms": 30000,
        "heartbeat_interval_ms": 3000,
        "max_poll_records": 500,
        "max_poll_interval_ms": 300000,
    }
    
    # Async Kafka Configuration (for aiokafka)
    AIOKAFKA_PRODUCER_CONFIG: dict = {
        "bootstrap_servers": None,  # Will be set from KAFKA_BOOTSTRAP_SERVERS
        "value_serializer": lambda v: v.encode('utf-8') if isinstance(v, str) else v,
        "key_serializer": lambda k: k.encode('utf-8') if isinstance(k, str) else k,
        "acks": "all",
        "request_timeout_ms": 30000,
        "compression_type": "gzip",
    }
    
    AIOKAFKA_CONSUMER_CONFIG: dict = {
        "bootstrap_servers": None,  # Will be set from KAFKA_BOOTSTRAP_SERVERS
        "group_id": "planbook-fastapi-async-group",
        "auto_offset_reset": "latest",
        "enable_auto_commit": True,
        "auto_commit_interval_ms": 1000,
        "session_timeout_ms": 30000,
        "heartbeat_interval_ms": 3000,
        "max_poll_records": 500,
    }
    
    # Connection Settings
    KAFKA_CONNECTION_TIMEOUT: int = int(os.getenv("KAFKA_CONNECTION_TIMEOUT", "10"))
    KAFKA_REQUEST_TIMEOUT: int = int(os.getenv("KAFKA_REQUEST_TIMEOUT", "30"))
    
    # Topic Management
    KAFKA_AUTO_CREATE_TOPICS: bool = os.getenv("KAFKA_AUTO_CREATE_TOPICS", "True").lower() == "true"
    KAFKA_TOPIC_PARTITIONS: int = int(os.getenv("KAFKA_TOPIC_PARTITIONS", "3"))
    KAFKA_TOPIC_REPLICATION_FACTOR: int = int(os.getenv("KAFKA_TOPIC_REPLICATION_FACTOR", "1"))
    
    # Message Settings
    KAFKA_MAX_MESSAGE_SIZE: int = int(os.getenv("KAFKA_MAX_MESSAGE_SIZE", "1048576"))  # 1MB
    
    # Security Settings (if needed)
    KAFKA_SECURITY_PROTOCOL: Optional[str] = os.getenv("KAFKA_SECURITY_PROTOCOL")
    KAFKA_SASL_MECHANISM: Optional[str] = os.getenv("KAFKA_SASL_MECHANISM")
    KAFKA_SASL_USERNAME: Optional[str] = os.getenv("KAFKA_SASL_USERNAME")
    KAFKA_SASL_PASSWORD: Optional[str] = os.getenv("KAFKA_SASL_PASSWORD")
    
    def __post_init__(self):
        """Initialize configuration after creation"""
        # Set bootstrap servers for all configs
        servers = self.KAFKA_BOOTSTRAP_SERVERS.split(",")
        
        self.KAFKA_PRODUCER_CONFIG["bootstrap_servers"] = servers
        self.KAFKA_CONSUMER_CONFIG["bootstrap_servers"] = servers
        self.AIOKAFKA_PRODUCER_CONFIG["bootstrap_servers"] = servers
        self.AIOKAFKA_CONSUMER_CONFIG["bootstrap_servers"] = servers
        
        # Add security settings if provided
        if self.KAFKA_SECURITY_PROTOCOL:
            security_config = {
                "security_protocol": self.KAFKA_SECURITY_PROTOCOL,
            }
            
            if self.KAFKA_SASL_MECHANISM:
                security_config.update({
                    "sasl_mechanism": self.KAFKA_SASL_MECHANISM,
                    "sasl_plain_username": self.KAFKA_SASL_USERNAME,
                    "sasl_plain_password": self.KAFKA_SASL_PASSWORD,
                })
            
            # Apply security config to all configurations
            self.KAFKA_PRODUCER_CONFIG.update(security_config)
            self.KAFKA_CONSUMER_CONFIG.update(security_config)
            self.AIOKAFKA_PRODUCER_CONFIG.update(security_config)
            self.AIOKAFKA_CONSUMER_CONFIG.update(security_config)
    
    class Config:
        case_sensitive = True


# Create global kafka settings instance
kafka_settings = KafkaSettings()

# Initialize configurations
kafka_settings.__post_init__()


def get_kafka_servers() -> List[str]:
    """Get list of Kafka bootstrap servers"""
    return kafka_settings.KAFKA_BOOTSTRAP_SERVERS.split(",")


def get_topic_name() -> str:
    """Get the main topic name"""
    return kafka_settings.KAFKA_TOPIC_NAME


def get_producer_config() -> dict:
    """Get producer configuration"""
    return kafka_settings.KAFKA_PRODUCER_CONFIG.copy()


def get_consumer_config() -> dict:
    """Get consumer configuration"""
    return kafka_settings.KAFKA_CONSUMER_CONFIG.copy()


def get_aiokafka_producer_config() -> dict:
    """Get async producer configuration"""
    return kafka_settings.AIOKAFKA_PRODUCER_CONFIG.copy()


def get_aiokafka_consumer_config() -> dict:
    """Get async consumer configuration"""
    return kafka_settings.AIOKAFKA_CONSUMER_CONFIG.copy()
