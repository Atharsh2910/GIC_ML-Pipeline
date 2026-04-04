"""
AI Agent Configuration
Settings for autonomous multi-agent system
"""

# Agent System Configuration
AGENT_SYSTEM_CONFIG = {
    'execution_mode': 'async',  # 'async' or 'sequential'
    'max_retries': 3,
    'timeout_seconds': 300,
    'enable_logging': True,
    'log_level': 'INFO'
}

# LangChain & LLM Configuration
LANGCHAIN_CONFIG = {
    'llm_provider': 'groq',  # 'groq', 'openai', 'anthropic', 'ollama'
    'model_name': 'llama-3.3-70b-versatile',
    'temperature': 0.1,
    'max_tokens': 2000,
    'top_p': 0.9,
    'api_key_env': 'GROQ_API_KEY',
}

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    'provider': 'pinecone',  # 'chromadb' or 'pinecone'
    'collection_name': 'gigshield_knowledge',
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'chunk_size': 500,
    'chunk_overlap': 50,
    # Chroma returns L2 distance; smaller = more similar
    'retrieval_config': {
        'top_k': 5,
        'max_l2_distance': 1.25,
        'include_metadata': True,
    },
    'pinecone': {
        # Env: PINECONE_API_KEY, PINECONE_HOST (serverless)
        # PINECONE_INDEX_NAME: must match the index name in the Pinecone console so describe_index can read dimension
        #   (defaults to gigshield-index if unset — wrong name => fallback to MiniLM 384 and dimension errors).
        # Or set PINECONE_INDEX_DIMENSION=1024 / GIGSHIELD_EMBEDDING_MODEL=intfloat/e5-large-v2 explicitly.
        'index_name': 'devtrails',
        'dimension': 384,
        'pinecone_index_dimension': 1024
    }
}

# MCP (Model Context Protocol) Configuration
MCP_CONFIG = {
    'weather_api': {
        'provider': 'openweathermap',
        'base_url': 'https://api.openweathermap.org/data/2.5',
        'endpoints': {
            'current': '/weather',
            'forecast': '/forecast',
            'alerts': '/alerts'
        },
        'polling_interval': 300,  # seconds
        'retry_attempts': 3
    },
    'news_api': {
        'provider': 'newsapi',
        'base_url': 'https://newsapi.org/v2',
        'endpoints': {
            'top_headlines': '/top-headlines',
            'everything': '/everything'
        },
        'polling_interval': 600,
        'keywords': ['strike', 'curfew', 'rally', 'protest', 'fuel shortage', 
                     'network outage', 'cyclone', 'weather alert']
    },
    'platform_api': {
        'endpoints': {
            'inventory': '/api/v1/inventory',
            'workers': '/api/v1/workers',
            'activity': '/api/v1/activity',
            'orders': '/api/v1/orders'
        },
        'polling_interval': 60,
        'batch_size': 1000
    }
}

# Monitor Agent Configuration
MONITOR_AGENT_CONFIG = {
    'name': 'MonitorAgent',
    'description': 'Continuously monitors external data sources for trigger conditions',
    'monitoring_frequency': 60,  # seconds
    'trigger_thresholds': {
        'rainfall_cm': 5.0,
        'temperature_high': 45.0,
        'temperature_low': 5.0,
        'cyclone_alert': 1,
        'network_outage_hours': 2,
        'fuel_shortage_severity': 0.3
    },
    'alert_channels': ['log', 'database', 'webhook']
}

# Validation Agent Configuration
VALIDATION_AGENT_CONFIG = {
    'name': 'ValidationAgent',
    'description': 'Validates trigger conditions and ensures data accuracy',
    'validation_rules': {
        'min_affected_workers': 5,
        'min_disruption_duration': 2,  # hours
        'geographic_radius_km': 5,
        'require_multiple_sources': True
    },
    'cross_validation_sources': ['weather_api', 'news_api', 'platform_api']
}

# Context Agent Configuration
CONTEXT_AGENT_CONFIG = {
    'name': 'ContextAgent',
    'description': 'Enriches context using RAG from vector database',
    'enable_llm_summary': False,
    'retrieval_config': {
        'top_k': 5,
        'similarity_threshold': 0.7,
        'include_metadata': True
    },
    'context_types': [
        'historical_disruptions',
        'similar_cases',
        'regional_patterns',
        'fraud_patterns',
        'policy_rules'
    ]
}

# Fraud Detection Agent Configuration
FRAUD_DETECTION_AGENT_CONFIG = {
    'name': 'FraudDetectionAgent',
    'description': 'Detects fraudulent patterns and anomalies',
    'detection_layers': [
        'gps_validation',
        'activity_validation',
        'peer_comparison',
        'behavioral_analysis',
        'device_fingerprinting'
    ],
    'fraud_scoring': {
        'weights': {
            'gps_spoofing': 0.25,
            'movement_anomaly': 0.20,
            'peer_deviation': 0.20,
            'behavior_inconsistency': 0.20,
            'device_sharing': 0.15
        },
        'threshold_reject': 0.8,
        'threshold_flag': 0.6
    }
}

# Rule Validation Agent Configuration
RULE_VALIDATION_AGENT_CONFIG = {
    'name': 'RuleValidationAgent',
    'description': 'Validates business rules and eligibility criteria',
    'validation_checks': [
        'cooling_period',
        'minimum_active_weeks',
        'premium_payment_status',
        'income_threshold',
        'employment_type',
        'disruption_severity'
    ],
    'rule_engine': {
        'strict_mode': True,
        'log_violations': True
    }
}

# Risk Scoring Agent Configuration
RISK_SCORING_AGENT_CONFIG = {
    'name': 'RiskScoringAgent',
    'description': 'Calculates comprehensive risk scores',
    'risk_components': {
        'income_risk': 0.25,
        'behavior_risk': 0.20,
        'fraud_risk': 0.20,
        'disruption_risk': 0.20,
        'location_risk': 0.15
    },
    'scoring_model': 'ensemble',  # Uses trained ML model
    'update_frequency': 'weekly'
}

# Decision Agent Configuration
DECISION_AGENT_CONFIG = {
    'name': 'DecisionAgent',
    'description': 'Orchestrates final decision making',
    'decision_matrix': {
        'auto_approve_threshold': 0.9,
        'auto_reject_threshold': 0.3,
        'manual_review_range': [0.3, 0.9]
    },
    'aggregation_method': 'weighted_average',
    'confidence_threshold': 0.7,
    'enable_explainability': True
}

# SQL Agent Configuration
SQL_AGENT_CONFIG = {
    'name': 'SQLAgent',
    'description': 'Handles database operations and data persistence',
    'database': {
        'type': 'postgresql',  # or 'sqlite'
        'host': 'localhost',
        'port': 5432,
        'database': 'gigshield',
        'schema': 'insurance'
    },
    'operations': {
        'batch_size': 1000,
        'max_retries': 3,
        'connection_pool_size': 10
    },
    'tables': {
        'workers': 'workers',
        'claims': 'claims',
        'premiums': 'premiums',
        'disruptions': 'disruptions',
        'fraud_logs': 'fraud_logs',
        'decisions': 'decisions'
    }
}

# Agent Communication Protocol
AGENT_COMMUNICATION = {
    'message_format': 'json',
    'include_timestamp': True,
    'include_trace_id': True,
    'priority_levels': ['critical', 'high', 'medium', 'low'],
    'timeout_handling': 'fallback_to_default'
}

# Orchestration Configuration
ORCHESTRATION_CONFIG = {
    'workflow_type': 'dag',  # Directed Acyclic Graph
    'parallel_agents': [
        'FraudDetectionAgent',
        'RuleValidationAgent', 
        'RiskScoringAgent'
    ],
    'sequential_agents': [
        'MonitorAgent',
        'ValidationAgent',
        'ContextAgent',
        'DecisionAgent',
        'SQLAgent'
    ],
    'error_handling': {
        'retry_failed_agents': True,
        'fallback_to_manual': False,
        'log_all_errors': True
    }
}

# RAG Knowledge Base Categories
RAG_KNOWLEDGE_CATEGORIES = {
    'insurance_policies': {
        'description': 'Insurance policy documents and rules',
        'embedding_strategy': 'hierarchical'
    },
    'historical_claims': {
        'description': 'Past claim cases and outcomes',
        'embedding_strategy': 'semantic'
    },
    'fraud_cases': {
        'description': 'Known fraud patterns and cases',
        'embedding_strategy': 'semantic'
    },
    'disruption_events': {
        'description': 'Historical disruption events and impacts',
        'embedding_strategy': 'temporal'
    },
    'regional_data': {
        'description': 'City-specific patterns and statistics',
        'embedding_strategy': 'geographic'
    }
}

# Agent Performance Metrics
AGENT_METRICS = {
    'track_latency': True,
    'track_accuracy': True,
    'track_throughput': True,
    'track_error_rate': True,
    'metrics_storage': 'database',
    'metrics_aggregation': 'hourly'
}