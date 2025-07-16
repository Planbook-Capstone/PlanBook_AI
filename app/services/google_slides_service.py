import logging
import threading
import os
import json
from typing import Dict, Any, List, Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.core.config import settings

logger = logging.getLogger(__name__)


def extract_text_from_shape(shape: Dict[str, Any]) -> str:
    if 'text' in shape and 'textElements' in shape['text']:
        parts = []
        for te in shape['text']['textElements']:
            text_run = te.get('textRun')
            if text_run:
                parts.append(text_run.get('content', '').strip())
        return ' '.join(parts).strip()
    return ''


def extract_text_style_from_shape(shape: Dict[str, Any], presentation: Dict[str, Any] = None) -> Dict[str, Any]:
    """Extract text style information including font size, family, etc."""
    text_styles = []
    placeholder_info = {}

    # Get placeholder information if available
    if 'placeholder' in shape:
        placeholder = shape['placeholder']
        placeholder_info = {
            'type': placeholder.get('type'),
            'parentObjectId': placeholder.get('parentObjectId')
        }

    if 'text' in shape and 'textElements' in shape['text']:
        for te in shape['text']['textElements']:
            text_run = te.get('textRun')
            if text_run:
                style = text_run.get('style', {})
                text_styles.append({
                    'content': text_run.get('content', '').strip(),
                    'style': style,
                    'hasStyle': bool(style)  # Track if style exists
                })

    # Return consolidated style info
    result = {
        'placeholder': placeholder_info if placeholder_info else None,
        'textElements': text_styles
    }

    # If we have styles, extract common properties
    styles_with_data = [ts for ts in text_styles if ts['hasStyle']]
    if styles_with_data:
        first_style = styles_with_data[0]['style']
        result.update({
            'fontSize': first_style.get('fontSize'),
            'fontFamily': first_style.get('fontFamily'),
            'bold': first_style.get('bold'),
            'italic': first_style.get('italic'),
            'underline': first_style.get('underline'),
            'foregroundColor': first_style.get('foregroundColor'),
            'backgroundColor': first_style.get('backgroundColor')
        })
    else:
        # No explicit styles found - try to get from parent placeholder
        inherited_style = {}
        if presentation and placeholder_info.get('parentObjectId'):
            inherited_style = find_parent_placeholder_style(
                presentation,
                placeholder_info['parentObjectId']
            )

        # Chỉ lấy giá trị thực từ API, không tạo default
        result.update({
            'fontSize': inherited_style.get('fontSize'),
            'fontFamily': inherited_style.get('fontFamily'),
            'bold': inherited_style.get('bold'),
            'italic': inherited_style.get('italic'),
            'underline': inherited_style.get('underline'),
            'foregroundColor': inherited_style.get('foregroundColor'),
            'backgroundColor': inherited_style.get('backgroundColor'),
            'note': 'Styles inherited from placeholder parent' if inherited_style else 'No explicit styles found in placeholder parent'
        })

    return result


def find_parent_placeholder_style(presentation: Dict[str, Any], parent_object_id: str) -> Dict[str, Any]:
    """Find and extract text style from parent placeholder"""

    def extract_style_from_element(element):
        """Extract style from a single element"""
        if 'shape' not in element:
            return {}

        shape = element['shape']
        if 'text' not in shape or 'textElements' not in shape['text']:
            return {}

        # Collect all styles from text elements and merge them
        merged_style = {}
        for te in shape['text']['textElements']:
            text_run = te.get('textRun')
            if text_run and 'style' in text_run:
                style = text_run['style']
                if style:  # Non-empty style
                    # Merge styles, prioritizing properties that exist
                    for key, value in style.items():
                        if key not in merged_style:
                            merged_style[key] = value

        return merged_style

    # Search in layouts first
    for layout in presentation.get('layouts', []):
        for element in layout.get('pageElements', []):
            if element.get('objectId') == parent_object_id:
                style = extract_style_from_element(element)
                if style:
                    return style

    # Search in masters if not found in layouts
    for master in presentation.get('masters', []):
        for element in master.get('pageElements', []):
            if element.get('objectId') == parent_object_id:
                style = extract_style_from_element(element)
                if style:
                    return style

    # Nếu không tìm thấy style từ parent placeholder, trả về empty dict
    return {}


class GoogleSlidesService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GoogleSlidesService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.slides_service = None
        self.drive_service = None
        self.credentials = None
        self._service_initialized = False
        self._initialized = True

    def _ensure_service_initialized(self):
        if not self._service_initialized:
            logger.info("🔄 GoogleSlidesService: First-time initialization triggered")
            self._initialize_service()
            self._service_initialized = True
            logger.info("✅ GoogleSlidesService: Initialization completed")

    def _initialize_service(self):
        """Khởi tạo Google Slides service với OAuth 2.0"""
        try:
            credentials_path = "google_client_Dat.json"
            token_path = "token.json"  # File lưu token sau khi authenticate

            if not os.path.exists(credentials_path):
                logger.warning("""
Google Slides service requires OAuth 2.0 Client credentials.
Please ensure google_client_Dat.json exists in the project root.
Service will be disabled.
                """)
                return

            # Scopes cần thiết
            SCOPES = [
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/presentations',
                'https://www.googleapis.com/auth/drive.file',
                'https://www.googleapis.com/auth/gmail.modify'
            ]

            creds = None

            # Kiểm tra xem đã có token chưa
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)

            # Nếu không có credentials hợp lệ, thực hiện OAuth flow
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    # Refresh token nếu expired
                    creds.refresh(Request())
                else:
                    # Thực hiện OAuth flow mới
                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_path, SCOPES
                    )
                    # Sử dụng local server để nhận callback
                    creds = flow.run_local_server()

                # Lưu token để sử dụng lần sau
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())

            self.credentials = creds

            # Tạo services
            self.slides_service = build('slides', 'v1', credentials=self.credentials)
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            logger.info("Google Slides service initialized with OAuth 2.0")

        except Exception as e:
            logger.error(f"Failed to initialize Google Slides service: {e}")
            self.slides_service = None
            self.drive_service = None

    def is_available(self) -> bool:
        self._ensure_service_initialized()
        return self.slides_service is not None and self.drive_service is not None

    async def copy_and_analyze_template(self, template_id: str, new_title: str) -> Dict[str, Any]:
        """
        Copy template và phân tích cấu trúc của bản sao (theo yêu cầu mới)

        Args:
            template_id: ID của Google Slides template gốc
            new_title: Tên cho file mới

        Returns:
            Dict chứa thông tin file đã copy và cấu trúc slides/elements
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Slides service not available"
            }

        try:
            # Bước 0: Validate template trước khi copy
            logger.info(f"Validating template {template_id} before copying...")
            validation_result = await self.validate_template(template_id)
            if not validation_result["success"]:
                return {
                    "success": False,
                    "error": f"Template validation failed: {validation_result['error']}"
                }

            # Bước 1: Copy template thành file mới ngay từ đầu
            logger.info(f"Copying template {template_id} to new file: {new_title}")
            copy_result = await self.copy_template(template_id, new_title)
            if not copy_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to copy template: {copy_result['error']}"
                }

            copied_presentation_id = copy_result["file_id"]
            logger.info(f"Template copied successfully. New presentation ID: {copied_presentation_id}")

            # Bước 2: Phân tích cấu trúc của bản sao (không phải template gốc)
            presentation = self.slides_service.presentations().get(
                presentationId=copied_presentation_id
            ).execute()

            slides_info = []
            for slide in presentation.get('slides', []):
                slide_info = {
                    "slideId": slide.get("objectId"),
                    "elements": []
                }

                for element in slide.get('pageElements', []):
                    if 'shape' in element:
                        text = extract_text_from_shape(element['shape'])
                        

                        element_info = {
                            "objectId": element.get('objectId'),
                            "text": text,
                        
                        }

                        slide_info['elements'].append(element_info)

                slides_info.append(slide_info)

            return {
                "success": True,
                "original_template_id": template_id,
                "copied_presentation_id": copied_presentation_id,
                "presentation_title": presentation.get('title', 'Untitled'),
                "web_view_link": copy_result["web_view_link"],
                "slide_count": len(presentation.get('slides', [])),
                "slides": slides_info
            }

        except HttpError as e:
            logger.error(f"HTTP error in copy_and_analyze_template: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {e}"
            }
        except Exception as e:
            logger.error(f"Error in copy_and_analyze_template: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def validate_template(self, template_id: str) -> Dict[str, Any]:
        """
        Kiểm tra template có tồn tại và có thể truy cập được không

        Args:
            template_id: ID của Google Slides template

        Returns:
            Dict kết quả validation
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Slides service not available"
            }

        try:
            # Kiểm tra file có tồn tại và có thể truy cập
            logger.info(f"Validating template file: {template_id}")

            # Thử lấy metadata của file
            file_metadata = self.drive_service.files().get(
                fileId=template_id,
                fields="id,name,mimeType,capabilities"
            ).execute()

            # Kiểm tra mime type
            mime_type = file_metadata.get('mimeType', '')
            if mime_type != 'application/vnd.google-apps.presentation':
                logger.error(f"Invalid template mime type: {mime_type}")
                return {
                    "success": False,
                    "error": f"Template is not a Google Slides presentation (mime type: {mime_type})"
                }

            # Kiểm tra quyền truy cập
            capabilities = file_metadata.get('capabilities', {})
            if not capabilities.get('canCopy', False):
                logger.error(f"Cannot copy template: missing copy permission")
                return {
                    "success": False,
                    "error": "Missing permission to copy this template"
                }

            logger.info(f"✅ Template validation successful: {file_metadata.get('name')}")
            return {
                "success": True,
                "template_name": file_metadata.get('name'),
                "template_id": template_id
            }

        except HttpError as e:
            logger.error(f"HTTP error validating template {template_id}: {e}")
            if e.resp.status == 404:
                return {
                    "success": False,
                    "error": f"Template not found: {template_id}"
                }
            elif e.resp.status == 403:
                return {
                    "success": False,
                    "error": f"Permission denied: You don't have access to this template"
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP error validating template: {e}"
                }
        except Exception as e:
            logger.error(f"Error validating template {template_id}: {e}")
            return {
                "success": False,
                "error": f"Error validating template: {str(e)}"
            }

    async def analyze_template_structure(self, template_id: str) -> Dict[str, Any]:
        """
        Phân tích cấu trúc template Google Slides (lấy text trên các slide)
        DEPRECATED: Sử dụng copy_and_analyze_template thay thế

        Args:
            template_id: ID của Google Slides template

        Returns:
            Dict chứa thông tin slides, objectId và text đang hiển thị
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Slides service not available"
            }

        try:
            presentation = self.slides_service.presentations().get(
                presentationId=template_id
            ).execute()

            slides_info = []
            for slide in presentation.get('slides', []):
                slide_info = {
                    "slideId": slide.get("objectId"),
                    "elements": []
                }

                for element in slide.get('pageElements', []):
                    if 'shape' in element:
                        text = extract_text_from_shape(element['shape'])
                        element_info = {
                            "objectId": element.get('objectId'),
                            "text": text,
                              
                        }
                
                        slide_info['elements'].append(element_info)

                slides_info.append(slide_info)


            return {
                "success": True,
                "template_id": template_id,
                "title": presentation.get('title', 'Untitled'),
                "slide_count": len(presentation.get('slides', [])),
                "slides": slides_info
            }

        except HttpError as e:
            logger.error(f"HTTP error analyzing template {template_id}: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {e}"
            }
        except Exception as e:
            logger.error(f"Error analyzing template {template_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def copy_template(self, template_id: str, new_title: str) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Slides service not available"
            }

        # Retry logic for Google API calls
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to copy template {template_id} (attempt {attempt + 1}/{max_retries})")

                copied_file = self.drive_service.files().copy(
                    fileId=template_id,
                    body={'name': new_title}
                ).execute()

                # Set permissions for the copied file
                self.drive_service.permissions().create(
                    fileId=copied_file.get('id'),
                    body={
                        'type': 'anyone',
                        'role': 'writer'
                    },
                    fields='id',
                    supportsAllDrives=True,
                    sendNotificationEmail=False
                ).execute()

                logger.info(f"✅ Template copied successfully on attempt {attempt + 1}")
                return {
                    "success": True,
                    "file_id": copied_file.get('id'),
                    "name": copied_file.get('name'),
                    "web_view_link": f"https://docs.google.com/presentation/d/{copied_file.get('id')}/edit"
                }

            except HttpError as e:
                logger.error(f"HTTP error copying template {template_id} (attempt {attempt + 1}): {e}")

                # Check if it's a retryable error
                if e.resp.status in [500, 502, 503, 504] and attempt < max_retries - 1:
                    logger.info(f"Retryable error, waiting {retry_delay} seconds before retry...")
                    import asyncio
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"HTTP error: {e}"
                    }

            except Exception as e:
                logger.error(f"Error copying template {template_id} (attempt {attempt + 1}): {e}")

                if attempt < max_retries - 1:
                    logger.info(f"Unexpected error, waiting {retry_delay} seconds before retry...")
                    import asyncio
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    return {
                        "success": False,
                        "error": str(e)
                    }

        # This should never be reached, but just in case
        return {
            "success": False,
            "error": f"Failed to copy template after {max_retries} attempts"
        }

    async def update_copied_presentation_content(
        self,
        presentation_id: str,
        slides_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Cập nhật nội dung vào presentation đã copy (theo quy trình mới)

        Args:
            presentation_id: ID của presentation đã copy
            slides_content: List nội dung slides từ LLM

        Returns:
            Dict kết quả cập nhật nội dung
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Slides service not available"
            }

        try:
            requests = []

            logger.info(f"📝 Updating presentation {presentation_id} with {len(slides_content)} slides")
            logger.info("🔄 Processing slides with reuse and duplicate support...")

            # Lấy thông tin presentation hiện tại
            presentation = self.slides_service.presentations().get(
                presentationId=presentation_id
            ).execute()

            logger.info(f"Current presentation has {len(presentation.get('slides', []))} slides")

            # Phase 1: Tạo tất cả slides trước
            create_requests = []
            slides_to_update = []
            slides_created = 0
            slides_updated = 0

            logger.info("🔄 Phase 1: Creating slides...")

            for slide_content in slides_content:
                slide_id = slide_content.get('slideId')
                action = slide_content.get('action', 'update')
                updates = slide_content.get('updates', {})

                if not slide_id or not updates:
                    logger.warning(f"Skipping slide with missing slideId or updates: {slide_content}")
                    continue

                logger.info(f"🔧 Processing slide {slide_id} with action '{action}' and {len(updates)} updates")

                # Xử lý tạo slide mới
                if action == 'create':
                    base_slide_id = slide_content.get('baseSlideId')

                    if base_slide_id:
                        # Tạo slide mới bằng cách duplicate slide base
                        create_requests.append({
                            'duplicateObject': {
                                'objectId': base_slide_id,
                                'objectIds': {
                                    base_slide_id: slide_id
                                }
                            }
                        })
                        slides_created += 1
                        logger.info(f"📄 Creating new slide {slide_id} based on {base_slide_id}")

                        # Lưu slide để update content sau
                        slides_to_update.append({
                            'slide_id': slide_id,
                            'base_slide_id': base_slide_id,
                            'updates': updates,
                            'action': action
                        })
                    else:
                        # Tạo slide trống mới
                        create_requests.append({
                            'createSlide': {
                                'objectId': slide_id,
                                'insertionIndex': len(presentation.get('slides', [])) + slides_created
                            }
                        })
                        slides_created += 1
                        logger.info(f"📄 Creating new blank slide {slide_id}")

                        # Lưu slide để update content sau
                        slides_to_update.append({
                            'slide_id': slide_id,
                            'base_slide_id': None,
                            'updates': updates,
                            'action': action
                        })
                else:
                    # Slide update - thêm vào list để xử lý
                    slides_to_update.append({
                        'slide_id': slide_id,
                        'base_slide_id': None,
                        'updates': updates,
                        'action': action
                    })

            # Thực thi create requests trước
            if create_requests:
                logger.info(f"⚡ Executing {len(create_requests)} create requests")

                self.slides_service.presentations().batchUpdate(
                    presentationId=presentation_id,
                    body={'requests': create_requests}
                ).execute()

                logger.info("✅ All slides created successfully")

                # Refresh presentation để lấy object IDs mới
                presentation = self.slides_service.presentations().get(
                    presentationId=presentation_id
                ).execute()
                logger.info("🔄 Refreshed presentation to get new object IDs")

                # Phase 1.5: Sắp xếp slides theo thứ tự đúng
                logger.info("🔄 Phase 1.5: Reordering slides to correct sequence...")
                await self._reorder_slides_by_sequence(presentation_id, slides_content)

            # Phase 2: Update content cho tất cả slides
            logger.info("🔄 Phase 2: Updating slide content...")

            update_requests = []

            for slide_info in slides_to_update:
                slide_id = slide_info['slide_id']
                base_slide_id = slide_info['base_slide_id']
                updates = slide_info['updates']
                action = slide_info['action']

                logger.info(f"📝 Updating content for slide: {slide_id}")

                # Map object IDs từ template sang slide mới (nếu là slide được duplicate)
                if action == 'create' and base_slide_id:
                    # Lấy mapping từ template elements sang new slide elements
                    object_id_mapping = self._get_object_id_mapping(
                        presentation, base_slide_id, slide_id
                    )

                    # Update content với object IDs mới
                    for old_object_id, new_content in updates.items():
                        new_object_id = object_id_mapping.get(old_object_id, old_object_id)

                        if not new_content:
                            continue

                        # Làm sạch nội dung
                        clean_content = str(new_content).strip()
                        if not clean_content:
                            continue

                        # Xóa nội dung cũ trước
                        update_requests.append({
                            'deleteText': {
                                'objectId': new_object_id,
                                'textRange': {
                                    'type': 'ALL'
                                }
                            }
                        })

                        # Thêm nội dung mới
                        update_requests.append({
                            'insertText': {
                                'objectId': new_object_id,
                                'text': clean_content,
                                'insertionIndex': 0
                            }
                        })

                        logger.debug(f"Mapped {old_object_id} -> {new_object_id}: {clean_content[:50]}...")
                else:
                    # Update slide hiện có với object IDs gốc
                    for element_id, new_content in updates.items():
                        if not new_content:
                            continue

                        # Làm sạch nội dung
                        clean_content = str(new_content).strip()
                        if not clean_content:
                            continue

                        # Xóa nội dung cũ trước
                        update_requests.append({
                            'deleteText': {
                                'objectId': element_id,
                                'textRange': {
                                    'type': 'ALL'
                                }
                            }
                        })

                        # Thêm nội dung mới
                        update_requests.append({
                            'insertText': {
                                'objectId': element_id,
                                'text': clean_content,
                                'insertionIndex': 0
                            }
                        })

                        logger.debug(f"Updated element {element_id} with content: {clean_content[:50]}...")

                if action == 'update':
                    slides_updated += 1

            # Thực thi update requests
            if update_requests:
                logger.info(f"⚡ Executing {len(update_requests)} update requests")

                # Chia requests thành các batch nhỏ để tránh timeout
                batch_size = 50
                for i in range(0, len(update_requests), batch_size):
                    batch_requests = update_requests[i:i + batch_size]
                    logger.info(f"Executing update batch {i//batch_size + 1}: {len(batch_requests)} requests")

                    self.slides_service.presentations().batchUpdate(
                        presentationId=presentation_id,
                        body={'requests': batch_requests}
                    ).execute()

                logger.info("✅ All content updates completed successfully")
            else:
                logger.warning("No content updates to execute")

            return {
                "success": True,
                "presentation_id": presentation_id,
                "slides_updated": slides_updated,
                "slides_created": slides_created,
                "total_slides_processed": len(slides_content),
                "requests_executed": len(requests)
            }

        except HttpError as e:
            logger.error(f"HTTP error creating slides for {presentation_id}: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {e}"
            }
        except Exception as e:
            logger.error(f"Error creating slides for {presentation_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _get_object_id_mapping(
        self,
        presentation: Dict[str, Any],
        template_slide_id: str,
        new_slide_id: str
    ) -> Dict[str, str]:
        """
        Tạo mapping từ object IDs của template slide sang object IDs của slide mới

        Args:
            presentation: Presentation data
            template_slide_id: ID của template slide
            new_slide_id: ID của slide mới được duplicate

        Returns:
            Dict mapping từ old object ID sang new object ID
        """
        try:
            # Tìm template slide
            template_slide = None
            new_slide = None

            for slide in presentation.get('slides', []):
                if slide.get('objectId') == template_slide_id:
                    template_slide = slide
                elif slide.get('objectId') == new_slide_id:
                    new_slide = slide

            if not template_slide or not new_slide:
                logger.warning(f"Could not find template slide {template_slide_id} or new slide {new_slide_id}")
                return {}

            # Tạo mapping dựa trên vị trí và type của elements
            mapping = {}
            template_elements = []
            new_elements = []

            # Lấy text elements từ template slide
            for element in template_slide.get('pageElements', []):
                if element.get('shape') and element.get('shape', {}).get('text'):
                    template_elements.append(element)

            # Lấy text elements từ new slide
            for element in new_slide.get('pageElements', []):
                if element.get('shape') and element.get('shape', {}).get('text'):
                    new_elements.append(element)

            # Map theo thứ tự (giả định elements được duplicate theo thứ tự)
            for i, template_elem in enumerate(template_elements):
                if i < len(new_elements):
                    template_id = template_elem.get('objectId')
                    new_id = new_elements[i].get('objectId')
                    if template_id and new_id:
                        mapping[template_id] = new_id
                        logger.debug(f"Mapped {template_id} -> {new_id}")

            logger.info(f"Created object ID mapping for {len(mapping)} elements")
            return mapping

        except Exception as e:
            logger.error(f"Error creating object ID mapping: {e}")
            return {}

    async def _reorder_slides_by_sequence(
        self,
        presentation_id: str,
        slides_content: List[Dict[str, Any]]
    ) -> None:
        """
        Sắp xếp lại slides theo thứ tự đúng dựa trên slide number trong tên

        Args:
            presentation_id: ID của presentation
            slides_content: List slides content với thứ tự mong muốn
        """
        try:
            logger.info("🔄 Reordering slides to correct sequence...")

            # Lấy thông tin presentation hiện tại
            presentation = self.slides_service.presentations().get(
                presentationId=presentation_id
            ).execute()

            current_slides = presentation.get('slides', [])
            logger.info(f"📊 Current slide order: {[slide.get('objectId') for slide in current_slides]}")

            # Tạo mapping từ slide ID sang thứ tự mong muốn
            desired_order = {}
            for i, slide_content in enumerate(slides_content):
                slide_id = slide_content.get('slideId')
                if slide_id:
                    desired_order[slide_id] = i

            logger.info(f"📋 Desired order mapping: {desired_order}")

            # Sắp xếp slides theo thứ tự mong muốn
            sorted_slides = sorted(current_slides, key=lambda slide: desired_order.get(slide.get('objectId'), 999))

            # Tạo requests để di chuyển slides
            reorder_requests = []
            for target_index, slide in enumerate(sorted_slides):
                slide_id = slide.get('objectId')
                current_index = next((i for i, s in enumerate(current_slides) if s.get('objectId') == slide_id), -1)

                if current_index != target_index and current_index != -1:
                    logger.info(f"📍 Moving slide {slide_id} from position {current_index} to {target_index}")
                    reorder_requests.append({
                        'updateSlidesPosition': {
                            'slideObjectIds': [slide_id],
                            'insertionIndex': target_index
                        }
                    })

            # Thực thi reorder requests
            if reorder_requests:
                logger.info(f"⚡ Executing {len(reorder_requests)} reorder requests")

                self.slides_service.presentations().batchUpdate(
                    presentationId=presentation_id,
                    body={'requests': reorder_requests}
                ).execute()

                logger.info("✅ Slides reordered successfully")

                # Verify new order
                updated_presentation = self.slides_service.presentations().get(
                    presentationId=presentation_id
                ).execute()
                new_order = [slide.get('objectId') for slide in updated_presentation.get('slides', [])]
                logger.info(f"📊 New slide order: {new_order}")
            else:
                logger.info("✅ Slides already in correct order")

        except Exception as e:
            logger.error(f"Error reordering slides: {e}")
            # Don't fail the entire process if reordering fails
            logger.warning("⚠️ Continuing without reordering...")

    async def delete_unused_slides(
        self,
        presentation_id: str,
        used_slide_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Xóa các slide không được sử dụng trong presentation

        Args:
            presentation_id: ID của presentation
            used_slide_ids: List các slide IDs đã được sử dụng

        Returns:
            Dict kết quả xóa slides
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Slides service not available"
            }

        try:
            # Lấy thông tin presentation hiện tại
            presentation = self.slides_service.presentations().get(
                presentationId=presentation_id
            ).execute()

            current_slides = presentation.get('slides', [])
            slides_to_delete = []

            # Tìm slides cần xóa (không có trong used_slide_ids)
            for slide in current_slides:
                slide_id = slide.get('objectId')
                if slide_id and slide_id not in used_slide_ids:
                    slides_to_delete.append(slide_id)

            if not slides_to_delete:
                logger.info("No unused slides to delete")
                return {
                    "success": True,
                    "slides_deleted": 0,
                    "message": "No unused slides found"
                }

            # Tạo requests để xóa slides
            requests = []
            for slide_id in slides_to_delete:
                requests.append({
                    'deleteObject': {
                        'objectId': slide_id
                    }
                })

            logger.info(f"Deleting {len(slides_to_delete)} unused slides: {slides_to_delete}")

            # Thực hiện xóa
            self.slides_service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': requests}
            ).execute()

            return {
                "success": True,
                "slides_deleted": len(slides_to_delete),
                "deleted_slide_ids": slides_to_delete
            }

        except HttpError as e:
            logger.error(f"HTTP error deleting unused slides: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {e}"
            }
        except Exception as e:
            logger.error(f"Error deleting unused slides: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def delete_template_slides(
        self,
        presentation_id: str,
        template_slide_ids: List[str],
        content_slide_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Xóa chỉ các template slides gốc, giữ lại slides có content

        Args:
            presentation_id: ID của presentation
            template_slide_ids: List các template slide IDs gốc
            content_slide_ids: List các slide IDs có content thực tế

        Returns:
            Dict kết quả xóa template slides
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Slides service not available"
            }

        try:
            logger.info(f"🧹 Starting template slides cleanup...")
            logger.info(f"   Template slides to check: {template_slide_ids}")
            logger.info(f"   Content slides to keep: {content_slide_ids}")

            # Lấy thông tin presentation hiện tại
            presentation = self.slides_service.presentations().get(
                presentationId=presentation_id
            ).execute()

            current_slides = presentation.get('slides', [])
            current_slide_ids = [slide.get('objectId') for slide in current_slides]

            logger.info(f"   Current slides in presentation: {current_slide_ids}")

            # Tìm template slides cần xóa (template gốc không có content)
            slides_to_delete = []
            for template_id in template_slide_ids:
                # Chỉ xóa nếu:
                # 1. Template slide vẫn tồn tại trong presentation
                # 2. Không có trong danh sách content slides
                if template_id in current_slide_ids and template_id not in content_slide_ids:
                    slides_to_delete.append(template_id)
                    logger.info(f"🗑️ Template slide marked for deletion: {template_id}")
                else:
                    logger.info(f"✅ Template slide kept (has content or not found): {template_id}")

            if not slides_to_delete:
                logger.info("ℹ️ No template slides to delete")
                return {
                    "success": True,
                    "slides_deleted": 0,
                    "message": "No template slides need to be deleted"
                }

            # Tạo requests để xóa template slides
            requests = []
            for slide_id in slides_to_delete:
                requests.append({
                    'deleteObject': {
                        'objectId': slide_id
                    }
                })

            logger.info(f"🗑️ Deleting {len(slides_to_delete)} template slides: {slides_to_delete}")

            # Thực hiện xóa
            self.slides_service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': requests}
            ).execute()

            logger.info(f"✅ Successfully deleted {len(slides_to_delete)} template slides")

            return {
                "success": True,
                "slides_deleted": len(slides_to_delete),
                "deleted_slide_ids": slides_to_delete,
                "kept_content_slides": content_slide_ids
            }

        except HttpError as e:
            logger.error(f"HTTP error deleting template slides: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {e}"
            }
        except Exception as e:
            logger.error(f"Error deleting template slides: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def delete_all_template_slides(
        self,
        presentation_id: str,
        template_slide_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Xóa TẤT CẢ template slides gốc (luồng mới)

        Args:
            presentation_id: ID của presentation
            template_slide_ids: List tất cả template slide IDs gốc cần xóa

        Returns:
            Dict kết quả xóa template slides
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Slides service not available"
            }

        try:
            logger.info(f"🧹 Starting template slides deletion process...")
            logger.info(f"   Template slides to delete: {template_slide_ids}")

            if not template_slide_ids:
                logger.info("ℹ️ No template slides to delete")
                return {
                    "success": True,
                    "slides_deleted": 0,
                    "message": "No template slides provided for deletion"
                }

            # Lấy thông tin presentation hiện tại
            presentation = self.slides_service.presentations().get(
                presentationId=presentation_id
            ).execute()

            current_slides = presentation.get('slides', [])
            current_slide_ids = [slide.get('objectId') for slide in current_slides]

            logger.info(f"   Current slides in presentation ({len(current_slides)} total): {current_slide_ids}")

            # Tìm template slides thực sự tồn tại để xóa
            slides_to_delete = []
            slides_not_found = []

            for template_id in template_slide_ids:
                if template_id in current_slide_ids:
                    slides_to_delete.append(template_id)
                    logger.info(f"🗑️ Template slide found and marked for deletion: {template_id}")
                else:
                    slides_not_found.append(template_id)
                    logger.warning(f"⚠️ Template slide not found in presentation: {template_id}")

            logger.info(f"📊 Deletion summary:")
            logger.info(f"   - Slides to delete: {len(slides_to_delete)}")
            logger.info(f"   - Slides not found: {len(slides_not_found)}")
            logger.info(f"   - Slides remaining after deletion: {len(current_slides) - len(slides_to_delete)}")

            if not slides_to_delete:
                logger.warning("⚠️ No template slides found to delete - they may have been already removed")
                return {
                    "success": True,
                    "slides_deleted": 0,
                    "message": "No template slides found in presentation",
                    "slides_not_found": slides_not_found
                }

            # Kiểm tra nếu xóa hết slides (Google Slides cần ít nhất 1 slide)
            remaining_slides = len(current_slides) - len(slides_to_delete)
            if remaining_slides <= 0:
                logger.error("❌ Cannot delete all slides - Google Slides requires at least 1 slide")
                return {
                    "success": False,
                    "error": "Cannot delete all slides - presentation must have at least 1 slide",
                    "slides_to_delete": slides_to_delete,
                    "current_slides_count": len(current_slides)
                }

            # Xóa từng slide một để tránh conflicts và dễ debug
            deleted_slides = []
            failed_slides = []

            # Xóa từ cuối lên đầu để tránh index issues
            slides_to_delete_reversed = list(reversed(slides_to_delete))
            logger.info(f"🗑️ Deleting {len(slides_to_delete)} template slides one by one in reverse order...")

            for i, slide_id in enumerate(slides_to_delete_reversed):
                try:
                    logger.info(f"   Deleting slide {i+1}/{len(slides_to_delete_reversed)}: {slide_id}")

                    # Xóa từng slide một
                    result = self.slides_service.presentations().batchUpdate(
                        presentationId=presentation_id,
                        body={'requests': [{
                            'deleteObject': {
                                'objectId': slide_id
                            }
                        }]}
                    ).execute()

                    deleted_slides.append(slide_id)
                    logger.info(f"   ✅ Successfully deleted slide: {slide_id}")

                except HttpError as e:
                    error_msg = str(e)
                    failed_slides.append({"slide_id": slide_id, "error": error_msg})
                    logger.error(f"   ❌ Failed to delete slide {slide_id}: {error_msg}")

                    # Nếu lỗi là "INVALID_REQUESTS" có thể slide không tồn tại
                    if "INVALID_REQUESTS" in error_msg:
                        logger.warning(f"   ⚠️ Slide {slide_id} may not exist or already deleted")

                except Exception as e:
                    error_msg = str(e)
                    failed_slides.append({"slide_id": slide_id, "error": error_msg})
                    logger.error(f"   ❌ Unexpected error deleting slide {slide_id}: {error_msg}")

            logger.info(f"✅ Deletion process completed:")
            logger.info(f"   - Successfully deleted: {len(deleted_slides)} slides")
            logger.info(f"   - Failed to delete: {len(failed_slides)} slides")

            if failed_slides:
                logger.warning(f"   - Failed slides: {[f['slide_id'] for f in failed_slides]}")

            return {
                "success": len(failed_slides) == 0,  # Success nếu không có slide nào fail
                "slides_deleted": len(deleted_slides),
                "deleted_slide_ids": deleted_slides,
                "failed_slides": failed_slides,
                "slides_not_found": slides_not_found,
                "remaining_slides": remaining_slides,
                "total_attempted": len(slides_to_delete)
            }

        except HttpError as e:
            logger.error(f"HTTP error deleting all template slides: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {e}"
            }
        except Exception as e:
            logger.error(f"Error deleting all template slides: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def debug_presentation_state(
        self,
        presentation_id: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Debug method để kiểm tra trạng thái presentation

        Args:
            presentation_id: ID của presentation
            description: Mô tả cho log

        Returns:
            Dict thông tin presentation
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Slides service not available"
            }

        try:
            logger.info(f"🔍 Debug presentation state: {description}")

            # Lấy thông tin presentation
            presentation = self.slides_service.presentations().get(
                presentationId=presentation_id
            ).execute()

            slides = presentation.get('slides', [])
            slide_ids = [slide.get('objectId') for slide in slides]

            logger.info(f"📊 Presentation info:")
            logger.info(f"   Title: {presentation.get('title', 'Untitled')}")
            logger.info(f"   Total slides: {len(slides)}")
            logger.info(f"   Slide IDs: {slide_ids}")

            # Chi tiết từng slide
            for i, slide in enumerate(slides):
                slide_id = slide.get('objectId')
                elements = slide.get('pageElements', [])
                logger.info(f"   Slide {i+1}: {slide_id} ({len(elements)} elements)")

            return {
                "success": True,
                "presentation_id": presentation_id,
                "title": presentation.get('title', 'Untitled'),
                "slide_count": len(slides),
                "slide_ids": slide_ids,
                "slides": slides
            }

        except HttpError as e:
            logger.error(f"HTTP error debugging presentation: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {e}"
            }
        except Exception as e:
            logger.error(f"Error debugging presentation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_presentation_details(self, presentation_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin chi tiết của Google Slides presentation

        Args:
            presentation_id: ID của presentation cần lấy thông tin

        Returns:
            Dict chứa thông tin chi tiết của presentation
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Slides service not available"
            }

        try:
            logger.info(f"🔍 Getting detailed information for presentation: {presentation_id}")

            # Lấy thông tin presentation từ Slides API
            presentation = self.slides_service.presentations().get(
                presentationId=presentation_id
            ).execute()

            # Lấy thông tin file từ Drive API để có thêm metadata
            file_metadata = self.drive_service.files().get(
                fileId=presentation_id,
                fields="id,name,createdTime,modifiedTime,webViewLink,owners,lastModifyingUser"
            ).execute()

            # Xử lý thông tin slides
            slides_info = []
            for i, slide in enumerate(presentation.get('slides', [])):
                slide_id = slide.get('objectId')

                # Xử lý thông tin elements trong slide
                elements_info = []
                for element in slide.get('pageElements', []):
                    element_type = "unknown"
                    element_text = None
                    element_properties = {}

                    # Xác định loại element và lấy thông tin tương ứng
                    if 'shape' in element:
                        element_type = "shape"
                        shape = element['shape']

                        if 'text' in shape:
                            element_text = extract_text_from_shape(shape)

                        # Lấy thông tin style chi tiết cho shape
                        shape_style = {}
                        if 'shapeProperties' in shape:
                            shape_props = shape['shapeProperties']

                            # Màu nền
                            if 'shapeBackgroundFill' in shape_props:
                                bg_fill = shape_props['shapeBackgroundFill']
                                if 'solidFill' in bg_fill and 'color' in bg_fill['solidFill']:
                                    color = bg_fill['solidFill']['color']
                                    if 'rgbColor' in color:
                                        rgb = color['rgbColor']
                                        shape_style['backgroundColor'] = {
                                            "red": rgb.get('red', 0),
                                            "green": rgb.get('green', 0),
                                            "blue": rgb.get('blue', 0)
                                        }
                                    elif 'themeColor' in color:
                                        shape_style['backgroundColor'] = {
                                            "themeColor": color['themeColor']
                                        }

                            # Đường viền
                            if 'outline' in shape_props:
                                outline = shape_props['outline']
                                outline_style = {}

                                if 'outlineFill' in outline and 'solidFill' in outline['outlineFill']:
                                    fill = outline['outlineFill']['solidFill']
                                    if 'color' in fill:
                                        color = fill['color']
                                        if 'rgbColor' in color:
                                            rgb = color['rgbColor']
                                            outline_style['color'] = {
                                                "red": rgb.get('red', 0),
                                                "green": rgb.get('green', 0),
                                                "blue": rgb.get('blue', 0)
                                            }
                                        elif 'themeColor' in color:
                                            outline_style['color'] = {
                                                "themeColor": color['themeColor']
                                            }

                                if 'weight' in outline:
                                    outline_style['weight'] = outline['weight']

                                if 'dashStyle' in outline:
                                    outline_style['dashStyle'] = outline['dashStyle']

                                if outline_style:
                                    shape_style['outline'] = outline_style

                        # Lấy thông tin style text
                        text_style = {}
                        text_alignment = {}

                        if 'text' in shape and 'textElements' in shape['text']:
                            text_elements = shape['text']['textElements']

                            # Lấy thông tin alignment từ paragraph style
                            if 'lists' in shape['text']:
                                text_alignment['lists'] = shape['text']['lists']

                            for text_elem in text_elements:
                                # Lấy paragraph style cho alignment
                                if 'paragraphMarker' in text_elem and 'style' in text_elem['paragraphMarker']:
                                    para_style = text_elem['paragraphMarker']['style']

                                    # Text alignment
                                    if 'alignment' in para_style:
                                        text_alignment['alignment'] = para_style['alignment']

                                    # Line spacing
                                    if 'lineSpacing' in para_style:
                                        text_alignment['lineSpacing'] = para_style['lineSpacing']

                                    # Space above/below
                                    if 'spaceAbove' in para_style:
                                        text_alignment['spaceAbove'] = para_style['spaceAbove']
                                    if 'spaceBelow' in para_style:
                                        text_alignment['spaceBelow'] = para_style['spaceBelow']

                                    # Indentation
                                    if 'indentStart' in para_style:
                                        text_alignment['indentStart'] = para_style['indentStart']
                                    if 'indentEnd' in para_style:
                                        text_alignment['indentEnd'] = para_style['indentEnd']
                                    if 'indentFirstLine' in para_style:
                                        text_alignment['indentFirstLine'] = para_style['indentFirstLine']

                                    # Direction
                                    if 'direction' in para_style:
                                        text_alignment['direction'] = para_style['direction']

                                if 'textRun' in text_elem and 'style' in text_elem['textRun']:
                                    style = text_elem['textRun']['style']

                                    # Font family
                                    if 'fontFamily' in style:
                                        text_style['fontFamily'] = style['fontFamily']

                                    # Font size
                                    if 'fontSize' in style:
                                        text_style['fontSize'] = style['fontSize']

                                    # Font weight
                                    if 'bold' in style:
                                        text_style['bold'] = style['bold']

                                    # Font style
                                    if 'italic' in style:
                                        text_style['italic'] = style['italic']

                                    # Underline
                                    if 'underline' in style:
                                        text_style['underline'] = style['underline']

                                    # Strikethrough
                                    if 'strikethrough' in style:
                                        text_style['strikethrough'] = style['strikethrough']

                                    # Text color
                                    if 'foregroundColor' in style and 'color' in style['foregroundColor']:
                                        color = style['foregroundColor']['color']
                                        if 'rgbColor' in color:
                                            rgb = color['rgbColor']
                                            text_style['color'] = {
                                                "red": rgb.get('red', 0),
                                                "green": rgb.get('green', 0),
                                                "blue": rgb.get('blue', 0)
                                            }
                                        elif 'themeColor' in color:
                                            text_style['color'] = {
                                                "themeColor": color['themeColor']
                                            }

                                    # Link
                                    if 'link' in style:
                                        text_style['link'] = style['link']

                                    break  # Chỉ lấy style của text element đầu tiên

                        element_properties = {
                            "shapeType": shape.get('shapeType', 'TEXT_BOX'),
                            "hasText": 'text' in shape,
                            "shapeStyle": shape_style,
                            "textStyle": text_style,
                            "textAlignment": text_alignment
                        }
                    elif 'image' in element:
                        element_type = "image"
                        image = element['image']

                        # Lấy thông tin style chi tiết cho image
                        image_style = {}
                        if 'imageProperties' in image:
                            img_props = image['imageProperties']

                            # Crop
                            if 'cropProperties' in img_props:
                                image_style['crop'] = img_props['cropProperties']

                            # Brightness
                            if 'brightness' in img_props:
                                image_style['brightness'] = img_props['brightness']

                            # Contrast
                            if 'contrast' in img_props:
                                image_style['contrast'] = img_props['contrast']

                            # Transparency
                            if 'transparency' in img_props:
                                image_style['transparency'] = img_props['transparency']

                            # Recolor
                            if 'recolor' in img_props:
                                image_style['recolor'] = img_props['recolor']

                            # Shadow
                            if 'shadow' in img_props:
                                image_style['shadow'] = img_props['shadow']

                            # Outline
                            if 'outline' in img_props:
                                image_style['outline'] = img_props['outline']

                        element_properties = {
                            "contentUrl": image.get('contentUrl', None),
                            "sourceUrl": image.get('sourceUrl', None),
                            "imageStyle": image_style
                        }
                    elif 'table' in element:
                        element_type = "table"
                        table = element['table']

                        # Lấy thông tin style chi tiết cho table
                        table_style = {}
                        if 'tableRows' in table:
                            table_rows = table['tableRows']
                            cells_data = []

                            for row_idx, row in enumerate(table_rows):
                                if 'tableCells' in row:
                                    for col_idx, cell in enumerate(row['tableCells']):
                                        cell_data = {
                                            "rowIndex": row_idx,
                                            "columnIndex": col_idx,
                                            "content": ""
                                        }

                                        # Lấy nội dung text của cell
                                        if 'text' in cell:
                                            cell_data["content"] = extract_text_from_shape({"text": cell['text']})

                                        # Lấy style của cell
                                        if 'tableCellProperties' in cell:
                                            cell_props = cell['tableCellProperties']
                                            cell_style = {}

                                            # Background color
                                            if 'tableCellBackgroundFill' in cell_props:
                                                bg_fill = cell_props['tableCellBackgroundFill']
                                                if 'solidFill' in bg_fill and 'color' in bg_fill['solidFill']:
                                                    color = bg_fill['solidFill']['color']
                                                    if 'rgbColor' in color:
                                                        rgb = color['rgbColor']
                                                        cell_style['backgroundColor'] = {
                                                            "red": rgb.get('red', 0),
                                                            "green": rgb.get('green', 0),
                                                            "blue": rgb.get('blue', 0)
                                                        }

                                            # Content alignment
                                            if 'contentAlignment' in cell_props:
                                                cell_style['contentAlignment'] = cell_props['contentAlignment']

                                            if cell_style:
                                                cell_data["style"] = cell_style

                                        cells_data.append(cell_data)

                            if cells_data:
                                table_style['cells'] = cells_data

                        element_properties = {
                            "rows": table.get('rows', 0),
                            "columns": table.get('columns', 0),
                            "hasTableCells": 'tableRows' in table,
                            "tableStyle": table_style
                        }
                    elif 'video' in element:
                        element_type = "video"
                        video = element['video']

                        # Lấy thông tin style chi tiết cho video
                        video_style = {}
                        if 'videoProperties' in video:
                            video_props = video['videoProperties']

                            # Autoplay
                            if 'autoPlay' in video_props:
                                video_style['autoPlay'] = video_props['autoPlay']

                            # Start/end time
                            if 'start' in video_props:
                                video_style['start'] = video_props['start']
                            if 'end' in video_props:
                                video_style['end'] = video_props['end']

                            # Mute
                            if 'mute' in video_props:
                                video_style['mute'] = video_props['mute']

                        element_properties = {
                            "videoProperties": video.get('videoProperties', {}),
                            "videoStyle": video_style,
                            "url": video.get('url', None),
                            "id": video.get('id', None),
                            "source": video.get('source', None)
                        }
                    elif 'line' in element:
                        element_type = "line"
                        line = element['line']

                        # Lấy thông tin style chi tiết cho line
                        line_style = {}
                        if 'lineProperties' in line:
                            line_props = line['lineProperties']

                            # Line weight
                            if 'weight' in line_props:
                                line_style['weight'] = line_props['weight']

                            # Dash style
                            if 'dashStyle' in line_props:
                                line_style['dashStyle'] = line_props['dashStyle']

                            # Line fill
                            if 'lineFill' in line_props:
                                line_fill = line_props['lineFill']
                                if 'solidFill' in line_fill and 'color' in line_fill['solidFill']:
                                    color = line_fill['solidFill']['color']
                                    if 'rgbColor' in color:
                                        rgb = color['rgbColor']
                                        line_style['color'] = {
                                            "red": rgb.get('red', 0),
                                            "green": rgb.get('green', 0),
                                            "blue": rgb.get('blue', 0)
                                        }

                            # Line end type
                            if 'endArrow' in line_props:
                                line_style['endArrow'] = line_props['endArrow']
                            if 'startArrow' in line_props:
                                line_style['startArrow'] = line_props['startArrow']

                        element_properties = {
                            "lineProperties": line.get('lineProperties', {}),
                            "lineStyle": line_style,
                            "lineType": line.get('lineType', None)
                        }

                    # Lấy thông tin transform chi tiết
                    transform_info = {}
                    if 'transform' in element:
                        transform = element['transform']

                        # Translation (vị trí)
                        transform_info['translateX'] = transform.get('translateX', 0)
                        transform_info['translateY'] = transform.get('translateY', 0)

                        # Scale (tỷ lệ)
                        transform_info['scaleX'] = transform.get('scaleX', 1.0)
                        transform_info['scaleY'] = transform.get('scaleY', 1.0)

                        # Shear (nghiêng)
                        transform_info['shearX'] = transform.get('shearX', 0)
                        transform_info['shearY'] = transform.get('shearY', 0)

                        # Unit (đơn vị)
                        transform_info['unit'] = transform.get('unit', 'EMU')
                    else:
                        # Default transform values
                        transform_info = {
                            'translateX': 0,
                            'translateY': 0,
                            'scaleX': 1.0,
                            'scaleY': 1.0,
                            'shearX': 0,
                            'shearY': 0,
                            'unit': 'EMU'
                        }

                    # Lấy thông tin size chi tiết
                    size_info = {}
                    if 'size' in element:
                        size = element['size']
                        if 'width' in size:
                            width = size['width']
                            size_info['width'] = {
                                'magnitude': width.get('magnitude', 0),
                                'unit': width.get('unit', 'EMU')
                            }
                        if 'height' in size:
                            height = size['height']
                            size_info['height'] = {
                                'magnitude': height.get('magnitude', 0),
                                'unit': height.get('unit', 'EMU')
                            }
                    else:
                        # Default size values
                        size_info = {
                            'width': {'magnitude': 0, 'unit': 'EMU'},
                            'height': {'magnitude': 0, 'unit': 'EMU'}
                        }

                    # Thêm thông tin element vào danh sách
                    elements_info.append({
                        "objectId": element.get('objectId'),
                        "type": element_type,
                        "text": element_text,
                        "position": {
                            "x": transform_info['translateX'],
                            "y": transform_info['translateY']
                        },
                        "size": size_info,
                        "transform": transform_info,
                        "properties": element_properties
                    })

                # Thêm thông tin slide vào danh sách
                slide_layout = None
                if 'slideProperties' in slide and 'layoutObjectId' in slide['slideProperties']:
                    slide_layout = slide['slideProperties']['layoutObjectId']

                slides_info.append({
                    "slide_id": slide_id,
                    "slide_index": i,
                    "layout": slide_layout,
                    "elements": elements_info,
                    "properties": {
                        "notesPage": "notesMaster" in slide,
                        "masterProperties": slide.get('slideProperties', {}).get('masterProperties', {})
                    }
                })

            # Tạo response
            return {
                "success": True,
                "presentation_id": presentation_id,
                "title": presentation.get('title', 'Untitled'),
                "slide_count": len(presentation.get('slides', [])),
                "slides": slides_info,
                "web_view_link": file_metadata.get('webViewLink'),
                "created_time": file_metadata.get('createdTime'),
                "last_modified_time": file_metadata.get('modifiedTime'),
                "owners": file_metadata.get('owners', []),
                "last_modified_by": file_metadata.get('lastModifyingUser', {})
            }

        except HttpError as e:
            logger.error(f"HTTP error getting presentation details: {e}")
            return {
                "success": False,
                "error": f"HTTP error: {e}"
            }
        except Exception as e:
            logger.error(f"Error getting presentation details: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def get_google_slides_service() -> GoogleSlidesService:
    return GoogleSlidesService()
