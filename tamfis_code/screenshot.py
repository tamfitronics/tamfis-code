"""
Screenshot capability for TAMFIS-CODE
Supports multiple backends: Codex-style (Playwright/Puppeteer) and Claude-style (PIL/OpenCV)
"""

import os
import sys
import json
import base64
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import tempfile
import time

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


@dataclass
class ScreenshotOptions:
    """Options for taking screenshots"""
    width: int = 1920
    height: int = 1080
    full_page: bool = True
    quality: int = 90
    format: str = "png"  # png, jpeg, webp
    selector: Optional[str] = None
    wait_for: Optional[float] = 2.0
    device_pixel_ratio: float = 1.0


class ScreenshotError(Exception):
    """Exception for screenshot failures"""
    pass


class ScreenshotTaker:
    """Main screenshot taking class with multiple backends"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(os.getcwd()) / "screenshots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = self._detect_backend()
    
    def _detect_backend(self) -> str:
        """Detect available backend"""
        if HAS_PLAYWRIGHT:
            return "playwright"
        elif HAS_PIL:
            return "pil"
        elif HAS_CV2:
            return "opencv"
        else:
            return "subprocess"
    
    def take_screenshot(
        self,
        url_or_path: str,
        filename: Optional[str] = None,
        options: Optional[ScreenshotOptions] = None,
        backend: Optional[str] = None
    ) -> Path:
        """
        Take a screenshot of a URL or file
        
        Args:
            url_or_path: URL or file path to screenshot
            filename: Output filename (auto-generated if None)
            options: Screenshot options
            backend: Force specific backend
        
        Returns:
            Path to the screenshot file
        """
        backend = backend or self.backend
        options = options or ScreenshotOptions()
        
        if filename is None:
            import uuid
            filename = f"screenshot_{uuid.uuid4().hex[:8]}.{options.format}"
        
        output_path = self.output_dir / filename
        
        if backend == "playwright":
            return self._take_screenshot_playwright(url_or_path, output_path, options)
        elif backend == "pil":
            return self._take_screenshot_pil(url_or_path, output_path, options)
        elif backend == "opencv":
            return self._take_screenshot_opencv(url_or_path, output_path, options)
        else:
            return self._take_screenshot_subprocess(url_or_path, output_path, options)
    
    def _take_screenshot_playwright(
        self, 
        url_or_path: str, 
        output_path: Path,
        options: ScreenshotOptions
    ) -> Path:
        """Take screenshot using Playwright (Codex-style)"""
        from playwright.async_api import async_playwright
        import asyncio
        
        async def _screenshot():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(
                    viewport={'width': options.width, 'height': options.height},
                    device_scale_factor=options.device_pixel_ratio
                )
                
                if url_or_path.startswith(('http://', 'https://')):
                    await page.goto(url_or_path, wait_until='networkidle')
                else:
                    # Serve local file
                    await page.goto(f'file://{os.path.abspath(url_or_path)}')
                
                if options.wait_for:
                    await page.wait_for_timeout(int(options.wait_for * 1000))
                
                if options.selector:
                    await page.wait_for_selector(options.selector)
                    element = await page.query_selector(options.selector)
                    await element.screenshot(path=str(output_path))
                else:
                    await page.screenshot(
                        path=str(output_path),
                        full_page=options.full_page,
                        quality=options.quality if options.format == "jpeg" else None
                    )
                
                await browser.close()
                return output_path
        
        return asyncio.run(_screenshot())
    
    def _take_screenshot_pil(
        self, 
        url_or_path: str, 
        output_path: Path,
        options: ScreenshotOptions
    ) -> Path:
        """Take screenshot using PIL (Claude-style)"""
        if not HAS_PIL:
            raise ScreenshotError("PIL not installed. Install with: pip install Pillow")
        
        # For local files, try to render them
        path = Path(url_or_path)
        if path.exists() and path.is_file():
            # Try to render as image
            try:
                img = Image.open(path)
                # Resize if needed
                if img.size[0] > options.width or img.size[1] > options.height:
                    img.thumbnail((options.width, options.height))
                img.save(output_path)
                return output_path
            except Exception as e:
                # If not an image, create a text representation
                return self._create_text_image(str(path), output_path, options)
        
        # If URL, try using web browser
        return self._create_text_image(url_or_path, output_path, options)
    
    def _take_screenshot_opencv(
        self, 
        url_or_path: str, 
        output_path: Path,
        options: ScreenshotOptions
    ) -> Path:
        """Take screenshot using OpenCV"""
        if not HAS_CV2:
            raise ScreenshotError("OpenCV not installed. Install with: pip install opencv-python")
        
        # For local images, use OpenCV
        path = Path(url_or_path)
        if path.exists() and path.is_file():
            img = cv2.imread(str(path))
            if img is not None:
                # Resize if needed
                height, width = img.shape[:2]
                if width > options.width or height > options.height:
                    scale = min(options.width/width, options.height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height))
                cv2.imwrite(str(output_path), img)
                return output_path
        
        # Fallback to PIL
        return self._take_screenshot_pil(url_or_path, output_path, options)
    
    def _take_screenshot_subprocess(
        self, 
        url_or_path: str, 
        output_path: Path,
        options: ScreenshotOptions
    ) -> Path:
        """Take screenshot using subprocess tools"""
        # Try different tools
        tools = [
            self._try_import_cmd,
            self._try_gnome_screenshot,
            self._try_scrot,
            self._try_chrome_headless,
            self._try_firefox_headless,
        ]
        
        for tool in tools:
            result = tool(url_or_path, output_path, options)
            if result:
                return output_path
        
        # Fallback: create a text screenshot
        return self._create_text_image(url_or_path, output_path, options)
    
    def _try_import_cmd(self, url_or_path: str, output_path: Path, options: ScreenshotOptions) -> bool:
        """Try using import (ImageMagick)"""
        try:
            cmd = ['import', '-window', 'root', str(output_path)]
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path.exists()
        except:
            return False
    
    def _try_gnome_screenshot(self, url_or_path: str, output_path: Path, options: ScreenshotOptions) -> bool:
        """Try using gnome-screenshot"""
        try:
            cmd = ['gnome-screenshot', '-f', str(output_path)]
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path.exists()
        except:
            return False
    
    def _try_scrot(self, url_or_path: str, output_path: Path, options: ScreenshotOptions) -> bool:
        """Try using scrot"""
        try:
            cmd = ['scrot', str(output_path)]
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path.exists()
        except:
            return False
    
    def _try_chrome_headless(self, url_or_path: str, output_path: Path, options: ScreenshotOptions) -> bool:
        """Try using Chrome headless"""
        if not url_or_path.startswith(('http://', 'https://')):
            return False
        try:
            cmd = [
                'google-chrome', '--headless', '--disable-gpu',
                f'--window-size={options.width},{options.height}',
                '--screenshot=' + str(output_path),
                url_or_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path.exists()
        except:
            return False
    
    def _try_firefox_headless(self, url_or_path: str, output_path: Path, options: ScreenshotOptions) -> bool:
        """Try using Firefox headless"""
        if not url_or_path.startswith(('http://', 'https://')):
            return False
        try:
            cmd = [
                'firefox', '--headless', '--screenshot', str(output_path),
                '--window-size', f'{options.width},{options.height}',
                url_or_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path.exists()
        except:
            return False
    
    def _create_text_image(
        self, 
        content: str, 
        output_path: Path, 
        options: ScreenshotOptions
    ) -> Path:
        """Create a simple text image when screenshot is not possible"""
        if not HAS_PIL:
            # Fallback: create a JSON/HTML file instead
            html_path = output_path.with_suffix('.html')
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Screenshot Preview</title></head>
            <body>
                <h1>Content Preview</h1>
                <pre>{content}</pre>
                <p><em>Screenshot not available. Install Playwright: pip install playwright && playwright install</em></p>
            </body>
            </html>
            """
            html_path.write_text(html_content)
            print(f"⚠️ Screenshot not available. Created HTML preview at {html_path}")
            return html_path
        
        img = Image.new('RGB', (options.width, options.height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Draw content
        lines = str(content).split('\n')
        y = 20
        for line in lines[:100]:
            draw.text((20, y), line[:120], fill='black', font=font)
            y += 18
        
        img.save(output_path)
        return output_path


# CLI Command for screenshot
async def screenshot_cli(url_or_path: str, **kwargs):
    """CLI wrapper for screenshot functionality"""
    taker = ScreenshotTaker()
    options = ScreenshotOptions(**kwargs)
    result = taker.take_screenshot(url_or_path, options=options)
    print(f"📸 Screenshot saved: {result}")
    return result


def screenshot_cli_sync(url_or_path: str, **kwargs):
    """Synchronous wrapper for CLI"""
    import asyncio
    return asyncio.run(screenshot_cli(url_or_path, **kwargs))


# Add to CLI
def add_screenshot_command(cli):
    """Add screenshot command to CLI"""
    @cli.command('screenshot')
    @click.argument('url_or_path')
    @click.option('--width', '-w', default=1920, help='Screenshot width')
    @click.option('--height', '-h', default=1080, help='Screenshot height')
    @click.option('--quality', '-q', default=90, help='JPEG quality (1-100)')
    @click.option('--format', '-f', default='png', help='Output format (png/jpeg/webp)')
    @click.option('--full-page', '-F', is_flag=True, help='Capture full page')
    @click.option('--output', '-o', help='Output filename')
    def screenshot_cmd(url_or_path, width, height, quality, format, full_page, output):
        """Take a screenshot of a URL or file"""
        taker = ScreenshotTaker()
        options = ScreenshotOptions(
            width=width,
            height=height,
            quality=quality,
            format=format,
            full_page=full_page,
        )
        try:
            result = taker.take_screenshot(
                url_or_path, 
                filename=output,
                options=options
            )
            click.echo(f"✅ Screenshot saved: {result}")
        except Exception as e:
            click.echo(f"❌ Screenshot failed: {e}")
    
    return cli
