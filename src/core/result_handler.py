"""
Result Handler

Manages saving, formatting, and exporting processing results in various formats.
"""

import json
import logging
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ResultHandler:
    """Handles saving and formatting of processing results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize result handler with configuration."""
        self.config = config
        self.output_config = config.get('output', {})
        self.results_dir = Path(self.output_config.get('results_directory', 'results'))
        self.save_enabled = self.output_config.get('save_results', True)
        self.export_format = self.output_config.get('export_format', 'json')
        
        # Create results directory
        if self.save_enabled:
            self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_result(self, result: Dict[str, Any]) -> Optional[str]:
        """Save a single processing result."""
        if not self.save_enabled:
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            blob_name_safe = self._sanitize_filename(result.get('blob_name', 'unknown'))
            
            if self.export_format.lower() == 'json':
                return await self._save_json_result(result, timestamp, blob_name_safe)
            elif self.export_format.lower() == 'csv':
                return await self._save_csv_result(result, timestamp, blob_name_safe)
            else:
                logger.warning(f"Unsupported export format: {self.export_format}")
                return await self._save_json_result(result, timestamp, blob_name_safe)
                
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            return None
    
    async def save_batch_results(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """Save batch processing results."""
        if not self.save_enabled or not results:
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if self.export_format.lower() == 'json':
                return await self._save_batch_json(results, timestamp)
            elif self.export_format.lower() == 'csv':
                return await self._save_batch_csv(results, timestamp)
            else:
                return await self._save_batch_json(results, timestamp)
                
        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")
            return None
    
    async def _save_json_result(self, result: Dict[str, Any], timestamp: str, blob_name: str) -> str:
        """Save result in JSON format."""
        filename = f"{timestamp}_{blob_name}_result.json"
        filepath = self.results_dir / filename
        
        # Create a clean result for saving
        clean_result = self._clean_result_for_export(result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Result saved: {filepath}")
        return str(filepath)
    
    async def _save_csv_result(self, result: Dict[str, Any], timestamp: str, blob_name: str) -> str:
        """Save result in CSV format."""
        filename = f"{timestamp}_{blob_name}_result.csv"
        filepath = self.results_dir / filename
        
        # Flatten result for CSV export
        flattened = self._flatten_result_for_csv(result)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if flattened:
                writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                writer.writeheader()
                writer.writerows(flattened)
        
        logger.info(f"📊 Result saved: {filepath}")
        return str(filepath)
    
    async def _save_batch_json(self, results: List[Dict[str, Any]], timestamp: str) -> str:
        """Save batch results in JSON format."""
        filename = f"{timestamp}_batch_results.json"
        filepath = self.results_dir / filename
        
        # Create summary and detailed results
        batch_data = {
            'batch_info': {
                'timestamp': timestamp,
                'total_files': len(results),
                'successful': len([r for r in results if not r.get('stages', {}).get('pipeline', {}).get('status') == 'failed']),
                'failed': len([r for r in results if r.get('stages', {}).get('pipeline', {}).get('status') == 'failed']),
            },
            'results': [self._clean_result_for_export(result) for result in results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Batch results saved: {filepath}")
        return str(filepath)
    
    async def _save_batch_csv(self, results: List[Dict[str, Any]], timestamp: str) -> str:
        """Save batch results in CSV format."""
        filename = f"{timestamp}_batch_results.csv"
        filepath = self.results_dir / filename
        
        # Flatten all results for CSV
        all_flattened = []
        for result in results:
            flattened = self._flatten_result_for_csv(result)
            all_flattened.extend(flattened)
        
        if all_flattened:
            # Get all unique fieldnames
            all_fieldnames = set()
            for row in all_flattened:
                all_fieldnames.update(row.keys())
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(all_fieldnames))
                writer.writeheader()
                writer.writerows(all_flattened)
        
        logger.info(f"📊 Batch results saved: {filepath}")
        return str(filepath)
    
    def _clean_result_for_export(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Clean result for export, removing temporary data."""
        clean_result = result.copy()
        
        # Remove temporary file paths
        if 'stages' in clean_result:
            for stage_name, stage_data in clean_result['stages'].items():
                if isinstance(stage_data, dict):
                    # Remove temporary paths
                    stage_data.pop('temp_path', None)
                    stage_data.pop('preprocessed_path', None)
        
        return clean_result
    
    def _flatten_result_for_csv(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten result structure for CSV export."""
        flattened_rows = []
        
        base_info = {
            'blob_name': result.get('blob_name', ''),
            'timestamp': result.get('timestamp', ''),
            'processing_time': result.get('processing_time', 0),
            'needs_human_review': result.get('needs_human_review', False),
            'document_type': result.get('extracted_data', {}).get('document_type', ''),
            'overall_confidence': result.get('extracted_data', {}).get('overall_confidence', 0),
        }
        
        # Add summary info
        summary = result.get('summary', {})
        base_info.update({
            'total_stages': summary.get('total_stages', 0),
            'completed_stages': summary.get('completed_stages', 0),
            'failed_stages': summary.get('failed_stages', 0),
            'success_rate': summary.get('success_rate', 0),
            'data_quality_score': summary.get('data_quality_score', 0),
        })
        
        # Add extracted fields
        extracted_data = result.get('extracted_data', {})
        for field_name, field_data in extracted_data.items():
            if isinstance(field_data, dict) and 'value' in field_data:
                base_info[f'field_{field_name}_value'] = field_data['value']
                base_info[f'field_{field_name}_confidence'] = field_data.get('confidence', 0)
            elif not field_name.endswith('_type') and not field_name.endswith('_confidence'):
                base_info[f'field_{field_name}'] = str(field_data)
        
        flattened_rows.append(base_info)
        return flattened_rows
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of saved results."""
        if not self.results_dir.exists():
            return {'total_files': 0, 'formats': []}
        
        result_files = list(self.results_dir.glob('*_result.*'))
        batch_files = list(self.results_dir.glob('*_batch_results.*'))
        
        return {
            'results_directory': str(self.results_dir),
            'total_result_files': len(result_files),
            'total_batch_files': len(batch_files),
            'formats': list(set(f.suffix[1:] for f in result_files + batch_files)),
            'latest_result': max(result_files, key=lambda f: f.stat().st_mtime).name if result_files else None,
            'latest_batch': max(batch_files, key=lambda f: f.stat().st_mtime).name if batch_files else None,
        }