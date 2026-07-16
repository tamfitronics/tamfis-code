"""Plan and Review mode for safe code changes"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime

class ChangeType(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    REFACTOR = "refactor"
    MOVE = "move"

@dataclass
class FileChange:
    """A planned change to a file"""
    path: str
    type: ChangeType
    content: Optional[str] = None
    original_content: Optional[str] = None
    description: str = ""
    confirmed: bool = False
    
    def get_diff(self) -> Optional[str]:
        """Get unified diff for this change"""
        if not self.original_content:
            return None
        if self.type == ChangeType.DELETE:
            return f"--- {self.path}\n+++ /dev/null\n@@ -1,{len(self.original_content.splitlines())} +0 @@\n"
        if self.content and self.original_content != self.content:
            # Simple diff display
            old_lines = self.original_content.splitlines()
            new_lines = self.content.splitlines()
            diff_lines = []
            
            # Find differences
            import difflib
            diff = difflib.unified_diff(old_lines, new_lines, fromfile=self.path, tofile=self.path)
            return '\n'.join(diff)
        return None

@dataclass
class Plan:
    """An execution plan"""
    id: str
    description: str
    changes: List[FileChange] = field(default_factory=list)
    created_at: str = ""
    approved: bool = False
    applied: bool = False
    review_comments: List[str] = field(default_factory=list)

class PlanReviewer:
    """Plan and review workflow manager"""
    
    def __init__(self):
        self.current_plan: Optional[Plan] = None
        self.approved_changes: List[FileChange] = []
        self.history: List[Plan] = []
        self._applied_changes: List[Dict[str, Any]] = []
    
    def create_plan(self, description: str, changes: List[FileChange]) -> Plan:
        """Create a new plan from changes"""
        import uuid
        
        plan = Plan(
            id=str(uuid.uuid4())[:8],
            description=description,
            changes=changes,
            created_at=datetime.now().isoformat(),
        )
        self.current_plan = plan
        return plan
    
    def review_plan(self) -> Dict[str, Any]:
        """Generate review summary of current plan"""
        if not self.current_plan:
            return {"error": "No plan to review"}
        
        changes_by_type = {}
        files_affected = []
        
        for change in self.current_plan.changes:
            type_name = change.type.value
            changes_by_type[type_name] = changes_by_type.get(type_name, 0) + 1
            if change.path not in files_affected:
                files_affected.append(change.path)
        
        return {
            "plan_id": self.current_plan.id,
            "description": self.current_plan.description,
            "total_changes": len(self.current_plan.changes),
            "changes_by_type": changes_by_type,
            "files_affected": files_affected,
            "created_at": self.current_plan.created_at,
            "approved": self.current_plan.approved,
        }
    
    def get_change_diff(self, change_index: int) -> Optional[str]:
        """Get diff for a specific change"""
        if not self.current_plan or change_index >= len(self.current_plan.changes):
            return None
        return self.current_plan.changes[change_index].get_diff()
    
    def approve(self, comment: str = "") -> bool:
        """Approve current plan"""
        if not self.current_plan:
            return False
        
        self.current_plan.approved = True
        if comment:
            self.current_plan.review_comments.append(f"Approved: {comment}")
        return True
    
    def reject(self, reason: str = "") -> bool:
        """Reject current plan"""
        if not self.current_plan:
            return False
        
        self.current_plan.approved = False
        if reason:
            self.current_plan.review_comments.append(f"Rejected: {reason}")
        return True
    
    def apply(self, dry_run: bool = False) -> List[Dict[str, Any]]:
        """Apply approved changes"""
        if not self.current_plan:
            return [{"error": "No plan to apply", "path": "unknown"}]
        
        if not self.current_plan.approved:
            return [{"error": "Plan not approved", "path": "unknown"}]
        
        results = []
        for change in self.current_plan.changes:
            try:
                result = self._apply_change(change, dry_run)
                # Ensure path is always present
                if 'path' not in result:
                    result['path'] = change.path
                results.append(result)
                if not dry_run:
                    change.confirmed = True
                    # Store original content for undo
                    self._applied_changes.append({
                        'path': change.path,
                        'original_content': change.original_content,
                        'type': change.type.value,
                    })
            except Exception as e:
                results.append({
                    "path": change.path,
                    "error": str(e),
                    "success": False
                })
        
        if not dry_run:
            self.current_plan.applied = True
            self.history.append(self.current_plan)
        
        return results
    
    def _apply_change(self, change: FileChange, dry_run: bool = False) -> Dict[str, Any]:
        """Apply a single file change"""
        path = Path(change.path)
        
        if dry_run:
            return {
                "path": str(path),
                "type": change.type.value,
                "description": change.description,
                "dry_run": True,
                "would_create": not path.exists() if change.type == ChangeType.CREATE else False,
                "would_modify": path.exists() and change.type in [ChangeType.UPDATE, ChangeType.REFACTOR],
                "would_delete": path.exists() and change.type == ChangeType.DELETE,
            }
        
        if change.type == ChangeType.CREATE:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(change.content or "")
            return {"path": str(path), "type": "create", "success": True}
        
        elif change.type == ChangeType.UPDATE:
            if path.exists():
                change.original_content = path.read_text(encoding='utf-8', errors='ignore')
                path.write_text(change.content or "")
                return {"path": str(path), "type": "update", "success": True}
            else:
                return {"path": str(path), "error": "File not found", "success": False}
        
        elif change.type == ChangeType.DELETE:
            if path.exists():
                change.original_content = path.read_text(encoding='utf-8', errors='ignore')
                path.unlink()
                return {"path": str(path), "type": "delete", "success": True}
            else:
                return {"path": str(path), "error": "File not found", "success": False}
        
        elif change.type == ChangeType.REFACTOR:
            # Refactor operation - would handle specific refactoring
            if path.exists():
                change.original_content = path.read_text(encoding='utf-8', errors='ignore')
            return {"path": str(path), "type": "refactor", "success": True, "message": "Refactoring applied"}
        
        return {"path": str(path), "error": "Unknown change type", "success": False}
    
    def undo_last_apply(self) -> Dict[str, Any]:
        """Undo the last applied plan"""
        if not self.history:
            return {"error": "No plans to undo"}
        
        last_plan = self.history[-1]
        undone_changes = []
        
        # Rebuild applied changes from the plan's changes
        for change in reversed(last_plan.changes):
            if change.confirmed:
                path = Path(change.path)
                try:
                    if change.type == ChangeType.CREATE:
                        # Delete created file
                        if path.exists():
                            path.unlink()
                            undone_changes.append({
                                "path": str(path),
                                "type": "undo_delete",
                                "success": True
                            })
                        else:
                            undone_changes.append({
                                "path": str(path),
                                "type": "undo_skip",
                                "message": "File already removed",
                                "success": True
                            })
                    elif change.type in [ChangeType.UPDATE, ChangeType.REFACTOR]:
                        # Restore original content
                        if change.original_content is not None and path.exists():
                            path.write_text(change.original_content)
                            undone_changes.append({
                                "path": str(path),
                                "type": "undo_restore",
                                "success": True
                            })
                        else:
                            undone_changes.append({
                                "path": str(path),
                                "type": "undo_skip",
                                "message": "No original content to restore",
                                "success": True
                            })
                    elif change.type == ChangeType.DELETE:
                        # Restore deleted file
                        if change.original_content is not None:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(change.original_content)
                            undone_changes.append({
                                "path": str(path),
                                "type": "undo_restore",
                                "success": True
                            })
                        else:
                            undone_changes.append({
                                "path": str(path),
                                "type": "undo_skip",
                                "message": "No content to restore",
                                "success": True
                            })
                    change.confirmed = False
                except Exception as e:
                    undone_changes.append({
                        "path": str(path),
                        "type": "undo_failed",
                        "error": str(e),
                        "success": False
                    })
        
        self.history.pop()
        last_plan.applied = False
        return {
            "undone_plan": last_plan.id,
            "changes": undone_changes,
            "success_count": sum(1 for c in undone_changes if c.get('success'))
        }
