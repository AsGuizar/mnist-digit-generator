import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import io
import base64

# Set page config
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .digit-button {
        display: inline-block;
        margin: 5px;
        padding: 10px 15px;
        background-color: #f0f2f6;
        border: 2px solid #667eea;
        border-radius: 50%;
        text-align: center;
        cursor: pointer;
        font-weight: bold;
        font-size: 18px;
    }
    .selected-digit {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Model definition (same as training script)
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=16, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim//2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim//2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        hidden = self.encoder(inputs)
        return self.mu_layer(hidden), self.logvar_layer(hidden)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        inputs = torch.cat([z, y], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, self.input_dim), y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Load model info
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        # Initialize model
        model = ConditionalVAE(
            input_dim=model_info['input_dim'],
            hidden_dim=model_info['hidden_dim'],
            latent_dim=model_info['latent_dim'],
            num_classes=model_info['num_classes']
        )
        
        # Load trained weights
        model.load_state_dict(torch.load('conditional_vae_mnist.pth', map_location='cpu'))
        model.eval()
        
        return model, model_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_digit_images(model, digit, num_samples=5):
    """Generate images for a specific digit"""
    with torch.no_grad():
        # Create one-hot encoding
        labels = torch.zeros(num_samples, 10)
        labels[:, digit] = 1
        
        # Sample from latent space with variations
        z = torch.randn(num_samples, model.latent_dim)
        # Add variations to ensure different looking digits
        for i in range(1, num_samples):
            z[i] = z[i] * (0.7 + 0.6 * torch.rand_like(z[i]))
        
        # Generate images
        generated = model.decode(z, labels)
        generated = generated.view(num_samples, 28, 28)
        
        # Convert to numpy and ensure proper range
        images = generated.cpu().numpy()
        images = np.clip(images, 0, 1)
        
    return images

def display_images(images, digit):
    """Display generated images in a grid"""
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    fig.suptitle(f'Generated Digit: {digit}', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray', interpolation='nearest')
        ax.set_title(f'Sample {i+1}', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

# Main app
def main():
    # Header
    st.title("🎨 MNIST Digit Generator")
    st.markdown("### Generate handwritten digits using a trained Conditional VAE")
    st.markdown("---")
    
    # Load model
    model, model_info = load_model()
    
    if model is None:
        st.error("❌ Could not load the trained model. Please ensure the model files are uploaded.")
        st.info("Required files: `conditional_vae_mnist.pth` and `model_info.pkl`")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Model info
    with st.expander("📊 Model Information"):
        if model_info:
            st.json(model_info)
            total_params = sum(p.numel() for p in model.parameters())
            st.write(f"**Total Parameters:** {total_params:,}")
    
    st.markdown("---")
    
    # Digit selection
    st.markdown("### Select a digit to generate (0-9):")
    
    # Create columns for digit buttons
    cols = st.columns(10)
    selected_digit = None
    
    for i, col in enumerate(cols):
        with col:
            if st.button(str(i), key=f"digit_{i}", help=f"Generate digit {i}"):
                selected_digit = i
    
    # Alternative: Use selectbox for mobile friendliness
    st.markdown("**Or use dropdown:**")
    selected_digit_dropdown = st.selectbox("Choose digit:", range(10), format_func=lambda x: f"Digit {x}")
    
    # Use dropdown selection if no button was pressed
    if selected_digit is None:
        selected_digit = selected_digit_dropdown
    
    st.markdown("---")
    
    # Generation section
    st.markdown(f"### Generating 5 images of digit: **{selected_digit}**")
    
    if st.button("🎯 Generate Images", type="primary"):
        with st.spinner("Generating images..."):
            try:
                # Generate images
                images = generate_digit_images(model, selected_digit, num_samples=5)
                
                # Display results
                st.markdown("### 🎉 Generated Images:")
                fig = display_images(images, selected_digit)
                st.pyplot(fig)
                
                # Additional info
                st.success(f"Successfully generated 5 unique images of digit {selected_digit}!")
                
                # Show individual images in columns for better mobile view
                st.markdown("#### Individual Images:")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        # Convert to PIL Image for better display
                        img_array = (images[i] * 255).astype(np.uint8)
                        img = Image.fromarray(img_array, mode='L')
                        # Resize for better visibility
                        img_resized = img.resize((112, 112), Image.NEAREST)
                        st.image(img_resized, caption=f"Sample {i+1}", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error generating images: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
        <p>🤖 Built with Streamlit | 🧠 Powered by Conditional VAE | 📊 Trained on MNIST Dataset</p>
        <p>This model was trained from scratch using PyTorch on Google Colab with T4 GPU</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()