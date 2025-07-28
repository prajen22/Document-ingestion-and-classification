import base64
import os

class BackgroundCSSGenerator:
    def __init__(self, img1_path, img2_path):
        self.img1_path = img1_path
        self.img2_path = img2_path

    def get_img_as_base64(self, file):
        # Ensure the file exists before opening
        if not os.path.exists(file):
            print(f"Error: Image file '{file}' not found.")
            return ""
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def generate_background_css(self):
        img1 = self.get_img_as_base64(self.img1_path)
        img2 = self.get_img_as_base64(self.img2_path)

        if not img1 or not img2:
            return "<style></style>"  # Return empty CSS if images are missing

        css = f"""
        <style>
        /* Apply background to the main content area */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/gif;base64,{img1}");
            background-size: cover;
            background-position: center;
        }}

        /* Apply background to the sidebar */
        [data-testid="stSidebarContent"] {{
            background-image: url("data:image/gif;base64,{img2}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Transparent header */
        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        /* Adjust toolbar position */
        [data-testid="stToolbar"] {{
            right: 2rem;
        }}
        </style>
        """
        return css
