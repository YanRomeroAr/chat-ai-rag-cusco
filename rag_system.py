"""
Sistema RAG principal que combina recuperación y generación
Este es el corazón del sistema que conecta Pinecone con OpenAI
"""

import openai
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from rag_config import rag_config
from pinecone_client import pinecone_client

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Estructura de respuesta del sistema RAG"""
    answer: str
    sources: List[Dict]
    confidence: float
    processing_time: float

class RAGSystem:
    """Sistema RAG completo para TUPA"""
    
    def __init__(self):
        """Inicializa el sistema RAG"""
        try:
            # Configurar OpenAI
            openai.api_key = rag_config.openai_api_key
            
            # Verificar conexiones
            if not pinecone_client:
                raise Exception("Cliente Pinecone no disponible")
            
            logger.info("✅ Sistema RAG inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema RAG: {e}")
            raise
    
    def _create_context(self, documents: List[Dict]) -> str:
        """
        Crea contexto para el prompt a partir de documentos recuperados
        
        Args:
            documents: Lista de documentos relevantes
            
        Returns:
            String con el contexto formateado
        """
        if not documents:
            return "No se encontraron documentos relevantes en la base de datos."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Incluir información de la fuente si está disponible
            source_info = ""
            if 'source' in doc.get('metadata', {}):
                source_info = f" (Fuente: {doc['metadata']['source']})"
            
            context_parts.append(
                f"Documento {i}{source_info}:\n{doc['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Crea el prompt para OpenAI incluyendo contexto y query
        
        Args:
            query: Pregunta del usuario
            context: Contexto de documentos relevantes
            
        Returns:
            Prompt formateado para OpenAI
        """
        prompt = f"""Eres un asistente especializado en el TUPA (Texto Único de Procedimientos Administrativos) del Gobierno Regional del Cusco.

Tu trabajo es responder preguntas sobre trámites administrativos basándote ÚNICAMENTE en la información proporcionada en el contexto.

INSTRUCCIONES IMPORTANTES:
1. Responde SOLO basándote en la información del contexto proporcionado
2. Si la información no está en el contexto, indica claramente que no tienes esa información específica
3. Sé claro, conciso y útil en tus respuestas
4. Menciona requisitos, plazos, costos y procedimientos cuando sea relevante
5. Si hay información sobre ubicaciones u horarios, inclúyela
6. Mantén un tono profesional pero amigable

CONTEXTO (Información oficial del TUPA):
{context}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:"""
        
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """
        Llama a OpenAI para generar la respuesta
        
        Args:
            prompt: Prompt completo con contexto y pregunta
            
        Returns:
            Respuesta generada por OpenAI
        """
        try:
            response = openai.chat.completions.create(
                model=rag_config.chat_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un asistente especializado en procedimientos administrativos del TUPA del Gobierno Regional del Cusco. Responde basándote únicamente en la información proporcionada."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=rag_config.max_tokens,
                temperature=rag_config.temperature,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error llamando a OpenAI: {e}")
            return "Lo siento, hubo un error procesando tu consulta. Por favor, intenta nuevamente."
    
    def _calculate_confidence(self, documents: List[Dict], query: str) -> float:
        """
        Calcula un score de confianza basado en la relevancia de documentos
        
        Args:
            documents: Documentos recuperados
            query: Query original
            
        Returns:
            Score de confianza entre 0 y 1
        """
        if not documents:
            return 0.0
        
        # Promedio de scores de similitud
        avg_score = sum(doc['score'] for doc in documents) / len(documents)
        
        # Ajustar basado en número de documentos encontrados
        doc_count_factor = min(len(documents) / rag_config.top_k_results, 1.0)
        
        # Score final
        confidence = (avg_score * 0.7) + (doc_count_factor * 0.3)
        
        return min(confidence, 1.0)
    
    def query(self, user_query: str) -> RAGResponse:
        """
        Procesa una consulta completa usando RAG
        
        Args:
            user_query: Pregunta del usuario
            
        Returns:
            RAGResponse con respuesta y metadatos
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Procesando consulta: {user_query[:100]}...")
            
            # 1. Recuperar documentos relevantes de Pinecone
            relevant_docs = pinecone_client.search_similar_documents(user_query)
            
            if not relevant_docs:
                logger.warning("⚠️ No se encontraron documentos relevantes")
                return RAGResponse(
                    answer="Lo siento, no encontré información específica sobre tu consulta en la base de datos del TUPA. ¿Podrías reformular tu pregunta o ser más específico?",
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
            
            # 2. Crear contexto para el prompt
            context = self._create_context(relevant_docs)
            
            # 3. Crear prompt completo
            prompt = self._create_prompt(user_query, context)
            
            # 4. Generar respuesta con OpenAI
            answer = self._call_openai(prompt)
            
            # 5. Calcular confianza
            confidence = self._calculate_confidence(relevant_docs, user_query)
            
            # 6. Preparar fuentes para mostrar al usuario
            sources = [
                {
                    'text': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                    'score': round(doc['score'], 3),
                    'metadata': doc.get('metadata', {})
                }
                for doc in relevant_docs[:3]  # Solo mostrar top 3 fuentes
            ]
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Consulta procesada en {processing_time:.2f}s con confianza {confidence:.2f}")
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error procesando consulta: {e}")
            return RAGResponse(
                answer="Lo siento, hubo un error procesando tu consulta. Por favor, intenta nuevamente.",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def health_check(self) -> Dict:
        """
        Verifica el estado de salud del sistema RAG
        
        Returns:
            Diccionario con información de salud
        """
        try:
            # Verificar Pinecone
            pinecone_stats = pinecone_client.get_index_stats()
            
            # Verificar OpenAI con una llamada simple
            test_response = openai.chat.completions.create(
                model=rag_config.chat_model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            
            return {
                "status": "healthy",
                "pinecone_vectors": pinecone_stats.get('total_vectors', 0),
                "openai_model": rag_config.chat_model,
                "embedding_model": rag_config.embedding_model,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

# Instancia global del sistema RAG
try:
    rag_system = RAGSystem()
    logger.info("✅ Sistema RAG disponible")
except Exception as e:
    logger.error(f"❌ Error inicializando sistema RAG: {e}")
    rag_system = None
