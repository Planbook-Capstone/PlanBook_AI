from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OMR Viewer"])

@router.get("/viewer", response_class=HTMLResponse)
async def omr_debug_viewer():
    """
    Trang web hiển thị debug images của OMR processing với AI Pipeline và Block Division
    """

    # Lấy danh sách debug images từ nhiều thư mục
    debug_dirs = {
        "OMR Processing": Path("data/grading/debug"),
        "All Markers": Path("data/grading/all_markers_debug"),
        "Block Division": Path("data/grading/block_division_debug")
    }

    all_debug_files = []
    debug_sections = {}

    for section_name, debug_dir in debug_dirs.items():
        section_files = []
        if debug_dir.exists():
            section_files = [f.name for f in sorted(debug_dir.glob("*.jpg"))]
        debug_sections[section_name] = {
            "files": section_files,
            "count": len(section_files),
            "dir": str(debug_dir)
        }
        all_debug_files.extend(section_files)

    # Backward compatibility - keep original debug_files for existing functionality
    debug_files = debug_sections.get("OMR Processing", {}).get("files", [])

    def get_image_url(section_name, filename):
        """Get the correct URL for different types of debug images"""
        if section_name == "OMR Processing":
            return f"/api/v1/omr_debug/debug_image/{filename}"
        elif section_name == "All Markers":
            return f"/api/v1/omr_debug/all_markers_image/{filename}"
        elif section_name == "Block Division":
            return f"/api/v1/omr_debug/block_division_image/{filename}"
        else:
            return f"/api/v1/omr_debug/debug_image/{filename}"
    
    # Tạo HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OMR Debug Viewer - PlanBook AI</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 3px solid #3498db;
                padding-bottom: 15px;
            }}
            .controls {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .btn {{
                background: #3498db;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 0 10px;
                transition: background 0.3s;
            }}
            .btn:hover {{
                background: #2980b9;
            }}
            .btn.success {{
                background: #27ae60;
            }}
            .btn.success:hover {{
                background: #229954;
            }}
            .btn.info {{
                background: #17a2b8;
            }}
            .btn.info:hover {{
                background: #138496;
            }}
            .btn.accent {{
                background: #6f42c1;
            }}
            .btn.accent:hover {{
                background: #5a32a3;
            }}
            .btn.danger {{
                background: #e74c3c;
            }}
            .btn.danger:hover {{
                background: #c0392b;
            }}
            .status {{
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                border-radius: 5px;
                font-weight: bold;
            }}
            .status.success {{
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .status.info {{
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }}
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }}
            .image-card {{
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                transition: transform 0.3s;
            }}
            .image-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .image-card img {{
                width: 100%;
                height: 200px;
                object-fit: cover;
                cursor: pointer;
            }}
            .image-card .title {{
                padding: 15px;
                font-weight: bold;
                color: #2c3e50;
                background: #ecf0f1;
                text-align: center;
            }}
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.9);
            }}
            .modal-content {{
                margin: auto;
                display: block;
                width: 90%;
                max-width: 1000px;
                max-height: 90%;
                object-fit: contain;
            }}
            .close {{
                position: absolute;
                top: 15px;
                right: 35px;
                color: #f1f1f1;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
            }}
            .close:hover {{
                color: #bbb;
            }}
            .loading {{
                text-align: center;
                color: #7f8c8d;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 OMR Debug Viewer + 🤖 AI Pipeline</h1>
            
            <div class="controls">
                <button class="btn success" onclick="processImage()">🚀 Xử lý ảnh test</button>
                <button class="btn success" onclick="processPipeline()">🤖 AI Pipeline</button>
                <button class="btn info" onclick="scanAllMarkers()">🔍 Scan All Markers</button>
                <button class="btn accent" onclick="divideBlocks()">📦 Divide Blocks</button>
                <button class="btn" onclick="refreshImages()">🔄 Làm mới</button>
                <button class="btn danger" onclick="clearImages()">🗑️ Xóa debug images</button>
            </div>
            
            <div id="status" class="status info">
                📊 Tìm thấy {len(all_debug_files)} debug images tổng cộng
            </div>

            <!-- Stats for different sections -->
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{debug_sections["OMR Processing"]["count"]}</div>
                    <div class="stat-label">OMR Processing</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{debug_sections["All Markers"]["count"]}</div>
                    <div class="stat-label">Marker Detection</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{debug_sections["Block Division"]["count"]}</div>
                    <div class="stat-label">Block Division</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(all_debug_files)}</div>
                    <div class="stat-label">Total Images</div>
                </div>
            </div>

            <!-- Sections for different types of debug images -->
            {"".join([f'''
            <div class="section" style="margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                <h2 style="color: #333; margin-bottom: 20px;">📁 {section_name} ({section_info["count"]} images)</h2>
                <div class="image-grid">
                    {"".join([f"""
                    <div class="image-card">
                        <img src="{get_image_url(section_name, filename)}"
                             alt="{filename}"
                             onclick="openModal(this.src, '{filename}')">
                        <div class="title">{filename}</div>
                    </div>
                    """ for filename in section_info["files"]])}
                </div>
                {f'<div class="loading">Chưa có {section_name.lower()} images.</div>' if not section_info["files"] else ''}
            </div>
            ''' for section_name, section_info in debug_sections.items() if section_info["count"] > 0])}

            {f'<div class="loading">Chưa có debug images. Nhấn các nút xử lý để bắt đầu.</div>' if not all_debug_files else ''}
        </div>
        
        <!-- Modal để hiển thị ảnh lớn -->
        <div id="imageModal" class="modal">
            <span class="close" onclick="closeModal()">&times;</span>
            <img class="modal-content" id="modalImage">
        </div>
        
        <script>
            function openModal(src, filename) {{
                document.getElementById('imageModal').style.display = 'block';
                document.getElementById('modalImage').src = src;
                document.getElementById('modalImage').alt = filename;
            }}
            
            function closeModal() {{
                document.getElementById('imageModal').style.display = 'none';
            }}
            
            async function processImage() {{
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '⏳ Đang xử lý ảnh...';
                statusDiv.className = 'status info';

                try {{
                    const response = await fetch('/api/v1/omr_debug/process_test_image', {{
                        method: 'POST'
                    }});

                    if (response.ok) {{
                        const result = await response.json();
                        statusDiv.innerHTML = `✅ Xử lý thành công! Student ID: ${{result.student_id}}, Test Code: ${{result.test_code}}, Answers: ${{result.total_answers}}`;
                        statusDiv.className = 'status success';

                        // Làm mới trang sau 2 giây
                        setTimeout(() => {{
                            window.location.reload();
                        }}, 2000);
                    }} else {{
                        throw new Error('Processing failed');
                    }}
                }} catch (error) {{
                    statusDiv.innerHTML = '❌ Lỗi xử lý ảnh: ' + error.message;
                    statusDiv.className = 'status error';
                }}
            }}

            async function processPipeline() {{
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '🤖 Đang xử lý với AI Pipeline...';
                statusDiv.className = 'status info';

                try {{
                    const response = await fetch('/api/v1/omr_debug/process_pipeline', {{
                        method: 'POST'
                    }});

                    if (response.ok) {{
                        const result = await response.json();
                        const section1 = result.results['Section I'];
                        const section2 = result.results['Section II'];
                        const section3 = result.results['Section III'];

                        statusDiv.innerHTML = `
                            ✅ AI Pipeline thành công!
                            📊 Section I: ${{section1.total_questions}} câu ABCD |
                            ✅ Section II: ${{section2.total_questions}} câu T/F |
                            🔢 Section III: ${{section3.total_questions}} câu Digits |
                            🖼️ Debug: ${{result.debug_info.total_debug_images}} images
                        `;
                        statusDiv.className = 'status success';

                        // Làm mới trang sau 3 giây
                        setTimeout(() => {{
                            window.location.reload();
                        }}, 3000);
                    }} else {{
                        throw new Error('Pipeline processing failed');
                    }}
                }} catch (error) {{
                    statusDiv.innerHTML = '❌ Lỗi AI Pipeline: ' + error.message;
                    statusDiv.className = 'status error';
                }}
            }}

            async function scanAllMarkers() {{
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '🔍 Đang quét tất cả marker...';
                statusDiv.className = 'status info';

                try {{
                    const response = await fetch('/api/v1/omr_debug/scan_all_markers', {{
                        method: 'POST'
                    }});

                    if (response.ok) {{
                        const result = await response.json();
                        const largeMarkers = result.markers.large_markers;
                        const smallMarkers = result.markers.small_markers;

                        statusDiv.innerHTML = `
                            ✅ Marker scanning thành công!
                            🔴 Large markers: ${{largeMarkers.count}} |
                            🟢 Small markers: ${{smallMarkers.count}} |
                            📊 Total: ${{largeMarkers.count + smallMarkers.count}} markers |
                            🖼️ Debug: ${{result.debug_info.total_debug_images}} images
                        `;
                        statusDiv.className = 'status success';

                        // Refresh images để hiển thị marker debug images
                        setTimeout(() => {{
                            refreshImages();
                        }}, 1000);

                        // Auto clear status after 5 seconds
                        setTimeout(() => {{
                            statusDiv.innerHTML = 'Sẵn sàng xử lý';
                            statusDiv.className = 'status info';
                        }}, 5000);
                    }} else {{
                        throw new Error('Marker scanning failed');
                    }}
                }} catch (error) {{
                    statusDiv.innerHTML = '❌ Lỗi marker scanning: ' + error.message;
                    statusDiv.className = 'status error';
                }}
            }}

            async function divideBlocks() {{
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '📦 Đang chia blocks dựa trên markers...';
                statusDiv.className = 'status info';

                try {{
                    // First, we need to create a FormData with the sample image
                    const response = await fetch('data/grading/sample.jpg');
                    const blob = await response.blob();
                    const formData = new FormData();
                    formData.append('file', blob, 'sample.jpg');

                    const divideResponse = await fetch('/api/v1/omr_debug/divide_blocks', {{
                        method: 'POST',
                        body: formData
                    }});

                    if (divideResponse.ok) {{
                        const result = await divideResponse.json();
                        const markerDetection = result.marker_detection;
                        const blockDivision = result.block_division;

                        statusDiv.innerHTML = `
                            ✅ Block division thành công!
                            🔴 Large markers: ${{markerDetection.large_markers_count}} |
                            🟢 Small markers: ${{markerDetection.small_markers_count}} |
                            📦 Total blocks: ${{blockDivision.total_blocks}} |
                            🏷️ Regions: ${{blockDivision.main_regions.length}} |
                            🖼️ Debug: ${{result.debug_info.total_debug_images}} images
                        `;
                        statusDiv.className = 'status success';

                        // Refresh images để hiển thị block division debug images
                        setTimeout(() => {{
                            refreshImages();
                        }}, 1000);

                        // Auto clear status after 5 seconds
                        setTimeout(() => {{
                            statusDiv.innerHTML = 'Sẵn sàng xử lý';
                            statusDiv.className = 'status info';
                        }}, 5000);
                    }} else {{
                        throw new Error('Block division failed');
                    }}
                }} catch (error) {{
                    statusDiv.innerHTML = '❌ Lỗi block division: ' + error.message;
                    statusDiv.className = 'status error';
                }}
            }}

            function refreshImages() {{
                window.location.reload();
            }}
            
            async function clearImages() {{
                if (confirm('Bạn có chắc muốn xóa tất cả debug images?')) {{
                    try {{
                        const response = await fetch('/api/v1/omr_debug/clear_debug', {{
                            method: 'DELETE'
                        }});
                        
                        if (response.ok) {{
                            window.location.reload();
                        }}
                    }} catch (error) {{
                        alert('Lỗi xóa images: ' + error.message);
                    }}
                }}
            }}
            
            // Đóng modal khi click bên ngoài
            window.onclick = function(event) {{
                const modal = document.getElementById('imageModal');
                if (event.target == modal) {{
                    closeModal();
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)
