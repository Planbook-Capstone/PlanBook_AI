from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OMR Viewer"])

@router.get("/viewer", response_class=HTMLResponse)
async def omr_debug_viewer(request: Request):
    """
    Trang web hiển thị debug images của OMR processing
    """
    
    # Lấy danh sách debug images
    debug_dir = Path("data/grading/debug")
    debug_files = []
    
    if debug_dir.exists():
        debug_files = [f.name for f in sorted(debug_dir.glob("*.jpg"))]
    
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
            <h1>🔍 OMR Debug Viewer</h1>
            
            <div class="controls">
                <button class="btn success" onclick="processImage()">🚀 Xử lý ảnh test</button>
                <button class="btn" onclick="refreshImages()">🔄 Làm mới</button>
                <button class="btn danger" onclick="clearImages()">🗑️ Xóa debug images</button>
            </div>
            
            <div id="status" class="status info">
                📊 Tìm thấy {len(debug_files)} debug images
            </div>
            
            <div id="imageGrid" class="image-grid">
                {"".join([f'''
                <div class="image-card">
                    <img src="/api/v1/omr_debug/debug_image/{filename}" 
                         alt="{filename}" 
                         onclick="openModal(this.src, '{filename}')">
                    <div class="title">{filename}</div>
                </div>
                ''' for filename in debug_files])}
            </div>
            
            {f'<div class="loading">Chưa có debug images. Nhấn "Xử lý ảnh test" để bắt đầu.</div>' if not debug_files else ''}
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
