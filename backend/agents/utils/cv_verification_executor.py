#!/usr/bin/env python3
"""
Computer Vision Verification Executor Script

This script provides a command-line interface for the Computer Vision Document Verification
system, allowing it to be called from Node.js API endpoints.

Usage:
    python cv_verification_executor.py --document <path> [--references <json_paths>] [--metadata <json_metadata>]

Features:
- Command-line interface for CV verification
- JSON input/output for seamless integration
- Error handling and logging
- Performance monitoring
- Batch processing support
"""

import asyncio
import argparse
import json
import logging
import sys
import os
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

# Add the agents directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from computer_vision_verifier import (
        ComputerVisionVerifier,
        AuthenticityReport,
        VerificationStatus,
        TamperingEvidence
    )
except ImportError as e:
    print(f"Error importing CV verification modules: {e}", file=sys.stderr)
    sys.exit(1)


def setup_logging():
    """Setup logging configuration for the executor."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(
                Path(__file__).parent.parent.parent.parent / 'logs' / 'cv_verification.log',
                mode='a'
            )
        ]
    )
    
    return logging.getLogger(__name__)


def serialize_authenticity_report(report: AuthenticityReport) -> Dict[str, Any]:
    """Convert AuthenticityReport to JSON-serializable dictionary."""
    try:
        return {
            'document_hash': report.document_hash,
            'verification_status': report.verification_status.value,
            'overall_confidence': float(report.overall_confidence),
            'forgery_probability': float(report.forgery_probability),
            'signature_authenticity': float(report.signature_authenticity),
            'tampering_evidence': [
                {
                    'tampering_type': evidence.tampering_type.value,
                    'confidence': float(evidence.confidence),
                    'location': list(evidence.location),
                    'description': evidence.description,
                    'technical_details': evidence.technical_details
                }
                for evidence in report.tampering_evidence
            ],
            'metadata_analysis': report.metadata_analysis,
            'image_forensics': report.image_forensics,
            'blockchain_hash': report.blockchain_hash,
            'verification_timestamp': report.verification_timestamp.isoformat(),
            'processing_time': float(report.processing_time)
        }
    except Exception as e:
        logger.error(f"Error serializing authenticity report: {e}")
        raise


def validate_file_path(file_path: str) -> bool:
    """Validate that file path exists and is readable."""
    try:
        path = Path(file_path)
        return path.exists() and path.is_file() and os.access(path, os.R_OK)
    except Exception:
        return False


def validate_image_file(file_path: str) -> bool:
    """Validate that file is a supported image format."""
    try:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.pdf'}
        path = Path(file_path)
        return path.suffix.lower() in allowed_extensions
    except Exception:
        return False


async def verify_single_document(document_path: str, 
                                reference_paths: List[str] = None,
                                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Verify a single document using computer vision.
    
    Args:
        document_path: Path to the document to verify
        reference_paths: Optional list of reference signature paths
        metadata: Optional metadata dictionary
        
    Returns:
        Serialized AuthenticityReport dictionary
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate document path
        if not validate_file_path(document_path):
            raise ValueError(f"Document file not found or not readable: {document_path}")
        
        if not validate_image_file(document_path):
            raise ValueError(f"Unsupported file format: {document_path}")
        
        # Validate reference paths if provided
        if reference_paths:
            for ref_path in reference_paths:
                if not validate_file_path(ref_path):
                    logger.warning(f"Reference file not found or not readable: {ref_path}")
                elif not validate_image_file(ref_path):
                    logger.warning(f"Unsupported reference file format: {ref_path}")
        
        logger.info(f"Starting CV verification for document: {document_path}")
        logger.info(f"Reference signatures: {len(reference_paths) if reference_paths else 0}")
        
        # Create verifier instance
        verifier = ComputerVisionVerifier()
        
        # Perform verification
        start_time = time.time()
        
        report = await verifier.verify_document(
            document_path=document_path,
            reference_signatures=reference_paths or [],
            metadata=metadata or {}
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"CV verification completed in {processing_time:.2f} seconds")
        logger.info(f"Verification status: {report.verification_status.value}")
        logger.info(f"Overall confidence: {report.overall_confidence:.3f}")
        
        # Serialize and return result
        result = serialize_authenticity_report(report)
        
        return {
            'success': True,
            'result': result,
            'execution_info': {
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'document_path': document_path,
                'reference_count': len(reference_paths) if reference_paths else 0
            }
        }
        
    except Exception as e:
        logger.error(f"CV verification failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'execution_info': {
                'timestamp': datetime.now().isoformat(),
                'document_path': document_path
            }
        }


async def verify_batch_documents(document_paths: List[str],
                               reference_paths_dict: Dict[str, List[str]] = None,
                               metadata_dict: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Verify multiple documents in batch.
    
    Args:
        document_paths: List of document paths to verify
        reference_paths_dict: Optional dict mapping document paths to reference paths
        metadata_dict: Optional dict mapping document paths to metadata
        
    Returns:
        Dictionary with batch verification results
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting batch CV verification for {len(document_paths)} documents")
        
        batch_start_time = time.time()
        results = []
        errors = []
        
        for i, doc_path in enumerate(document_paths):
            logger.info(f"Processing document {i+1}/{len(document_paths)}: {doc_path}")
            
            # Get references and metadata for this document
            references = reference_paths_dict.get(doc_path, []) if reference_paths_dict else []
            metadata = metadata_dict.get(doc_path, {}) if metadata_dict else {}
            
            try:
                # Verify single document
                result = await verify_single_document(doc_path, references, metadata)
                
                if result['success']:
                    results.append({
                        'document_path': doc_path,
                        'verification_result': result['result'],
                        'processing_time': result['execution_info']['processing_time']
                    })
                else:
                    errors.append({
                        'document_path': doc_path,
                        'error': result['error'],
                        'error_type': result['error_type']
                    })
                    
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {str(e)}")
                errors.append({
                    'document_path': doc_path,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        batch_end_time = time.time()
        total_processing_time = batch_end_time - batch_start_time
        
        logger.info(f"Batch verification completed in {total_processing_time:.2f} seconds")
        logger.info(f"Success: {len(results)}, Errors: {len(errors)}")
        
        return {
            'success': True,
            'batch_results': {
                'total_documents': len(document_paths),
                'successful_verifications': len(results),
                'failed_verifications': len(errors),
                'total_processing_time': total_processing_time,
                'average_processing_time': total_processing_time / len(document_paths) if document_paths else 0,
                'results': results,
                'errors': errors
            },
            'execution_info': {
                'timestamp': datetime.now().isoformat(),
                'batch_size': len(document_paths)
            }
        }
        
    except Exception as e:
        logger.error(f"Batch CV verification failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'execution_info': {
                'timestamp': datetime.now().isoformat(),
                'batch_size': len(document_paths) if document_paths else 0
            }
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Computer Vision Document Verification Executor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Verify single document
    python cv_verification_executor.py --document /path/to/document.pdf
    
    # Verify with reference signatures
    python cv_verification_executor.py --document /path/to/document.pdf --references '["ref1.jpg", "ref2.jpg"]'
    
    # Verify with metadata
    python cv_verification_executor.py --document /path/to/document.pdf --metadata '{"client_id": "12345"}'
    
    # Batch verification
    python cv_verification_executor.py --batch '["doc1.pdf", "doc2.pdf"]'
        """
    )
    
    # Single document verification
    parser.add_argument(
        '--document',
        type=str,
        help='Path to the document to verify'
    )
    
    parser.add_argument(
        '--references',
        type=str,
        help='JSON array of reference signature file paths'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        help='JSON object containing document metadata'
    )
    
    # Batch verification
    parser.add_argument(
        '--batch',
        type=str,
        help='JSON array of document paths for batch verification'
    )
    
    parser.add_argument(
        '--batch-references',
        type=str,
        help='JSON object mapping document paths to reference signature arrays'
    )
    
    parser.add_argument(
        '--batch-metadata',
        type=str,
        help='JSON object mapping document paths to metadata objects'
    )
    
    # Output options
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file path (default: stdout)'
    )
    
    parser.add_argument(
        '--pretty-json',
        action='store_true',
        help='Pretty-print JSON output'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all logging except errors'
    )
    
    return parser.parse_args()


def configure_logging(verbose: bool, quiet: bool) -> logging.Logger:
    """Configure logging based on command line options."""
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    
    return logging.getLogger(__name__)


async def main():
    """Main execution function."""
    global logger
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Configure logging
        logger = configure_logging(args.verbose, args.quiet)
        
        # Validate arguments
        if not args.document and not args.batch:
            logger.error("Either --document or --batch must be provided")
            sys.exit(1)
        
        if args.document and args.batch:
            logger.error("Cannot specify both --document and --batch")
            sys.exit(1)
        
        # Execute verification
        if args.document:
            # Single document verification
            logger.debug(f"Single document mode: {args.document}")
            
            # Parse references
            references = []
            if args.references:
                try:
                    references = json.loads(args.references)
                    if not isinstance(references, list):
                        raise ValueError("References must be a JSON array")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid references JSON: {e}")
                    sys.exit(1)
            
            # Parse metadata
            metadata = {}
            if args.metadata:
                try:
                    metadata = json.loads(args.metadata)
                    if not isinstance(metadata, dict):
                        raise ValueError("Metadata must be a JSON object")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid metadata JSON: {e}")
                    sys.exit(1)
            
            # Execute verification
            result = await verify_single_document(args.document, references, metadata)
            
        else:
            # Batch verification
            logger.debug("Batch verification mode")
            
            # Parse batch documents
            try:
                document_paths = json.loads(args.batch)
                if not isinstance(document_paths, list):
                    raise ValueError("Batch documents must be a JSON array")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid batch documents JSON: {e}")
                sys.exit(1)
            
            # Parse batch references
            references_dict = {}
            if args.batch_references:
                try:
                    references_dict = json.loads(args.batch_references)
                    if not isinstance(references_dict, dict):
                        raise ValueError("Batch references must be a JSON object")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid batch references JSON: {e}")
                    sys.exit(1)
            
            # Parse batch metadata
            metadata_dict = {}
            if args.batch_metadata:
                try:
                    metadata_dict = json.loads(args.batch_metadata)
                    if not isinstance(metadata_dict, dict):
                        raise ValueError("Batch metadata must be a JSON object")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid batch metadata JSON: {e}")
                    sys.exit(1)
            
            # Execute batch verification
            result = await verify_batch_documents(document_paths, references_dict, metadata_dict)
        
        # Output result
        if args.pretty_json:
            json_output = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            json_output = json.dumps(result, ensure_ascii=False)
        
        if args.output_file:
            try:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                logger.info(f"Results written to: {args.output_file}")
            except Exception as e:
                logger.error(f"Failed to write output file: {e}")
                sys.exit(1)
        else:
            print(json_output)
        
        # Exit with appropriate code
        if result.get('success', False):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Output error as JSON
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'execution_info': {
                'timestamp': datetime.now().isoformat()
            }
        }
        
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)


if __name__ == '__main__':
    # Initialize logger
    logger = setup_logging()
    
    # Run main function
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
