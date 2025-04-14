import base64
import logging
import requests
import plantuml


logger = logging.getLogger(__name__)

def generate_plantuml_image(plantuml_code):
    """Generate a PNG image from PlantUML code using a web service"""
    logger.info("Generating PlantUML image")
    
    # Make sure the PlantUML code has the required tags
    plantuml_code = plantuml_code.strip()
    if not plantuml_code.startswith('@startuml'):
        logger.debug("Adding @startuml tag")
        plantuml_code = '@startuml\n' + plantuml_code
    if not plantuml_code.endswith('@enduml'):
        logger.debug("Adding @enduml tag")
        plantuml_code = plantuml_code + '\n@enduml'
    
    # # Save processed PlantUML code for debugging
    # with open("log\processed_plantuml.txt", "w") as f:
    #     f.write(plantuml_code)
    
    # Try to generate the UML diagram image
    try:
        result = generate_plantuml_with_server(plantuml_code)
        if result:
            return result
    except Exception as e:
        logger.error(f"Error with UML generation: {str(e)}")
    
    # If method fails, create a simple text-based UML
    logger.error("UML generation failed, using fallback")
    return generate_fallback_uml_image()

def generate_plantuml_with_server(plantuml_code):
    """Generate PlantUML using plantuml.com server with proper encoding"""
    logger.info("Trying PlantUML generation with plantuml.com server")
    
    try:
        # Initialize PlantUML client with server URL
        plantuml_client = plantuml.PlantUML("http://www.plantuml.com/plantuml/")
        
        # Get the encoded URL using the library's method
        encoded_url = plantuml_client.get_url(plantuml_code)
        
        # Extract just the encoded part (after last slash)
        encoded = encoded_url.split('/')[-1]
        
        # Build the full URL
        url = f"https://www.plantuml.com/plantuml/png/{encoded}"
        logger.info(f"Fetching PlantUML image from: {url[:100]}...")
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200 and response.content:
            logger.info(f"Successfully generated PlantUML image ({len(response.content)} bytes)")
            return base64.b64encode(response.content).decode('utf-8')
        else:
            logger.warning(f"PlantUML server returned non-200 status: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error with plantuml.com server method: {str(e)}")
        return None

def generate_fallback_uml_image():
    """Generate a simple text-based fallback image when regular UML generation fails"""
    logger.info("Generating fallback UML image")
    
    # Create a simple SVG with text saying "UML Generation Failed"
    svg_content = """
    <svg xmlns="http://www.w3.org/2000/svg" width="300" height="100">
        <rect width="300" height="100" fill="#f8f9fa" />
        <text x="150" y="50" font-family="Arial" font-size="14" text-anchor="middle" fill="#dc3545">
            UML Generation Failed
        </text>
        <text x="150" y="70" font-family="Arial" font-size="12" text-anchor="middle" fill="#6c757d">
            Please check PlantUML syntax
        </text>
    </svg>
    """
    
    # Convert SVG to base64
    svg_bytes = svg_content.encode('utf-8')
    base64_svg = base64.b64encode(svg_bytes).decode('utf-8')
    
    # We return it as if it was a PNG because that's what the client expects
    return base64_svg