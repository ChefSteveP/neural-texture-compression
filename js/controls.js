const Controls = {
    setupMouseControls() {
        // Mouse down event - start tracking
        document.addEventListener('mousedown', function(event) {
            App.mouse.isDown = true;
            App.mouse.lastX = event.clientX;
            App.mouse.lastY = event.clientY;
        });
        
        // Mouse move event - calculate rotation if mouse is down
        document.addEventListener('mousemove', function(event) {
            if (!App.mouse.isDown) return;
            
            // Calculate the mouse movement delta
            const deltaX = event.clientX - App.mouse.lastX;
            const deltaY = event.clientY - App.mouse.lastY;

            App.rotation.y += deltaX * App.mouse.sensitivity;
            App.rotation.x += deltaY * App.mouse.sensitivity;

            // Update last position for next move event
            App.mouse.lastX = event.clientX;
            App.mouse.lastY = event.clientY;
        });
        
        // Mouse up event - stop tracking
        document.addEventListener('mouseup', function() {
            App.mouse.isDown = false;
        });
        
        // Mouse leave event - stop tracking if cursor leaves window
        document.addEventListener('mouseleave', function() {
            App.mouse.isDown = false;
        });
    },
      setupEventListeners(app) {
        // Helper function to safely add event listeners
        const addSafeListener = (id, event, handler) => {
          const element = document.getElementById(id);
          if (element) {
            element.addEventListener(event, handler);
          } else {
            console.warn(`Element with id '${id}' not found, skipping event listener`);
          }
        };
      
        // Texture upload listeners
        addSafeListener('albedo-input', 'change', function() {
          Controls.handleTextureUpload(this, 'albedo', app);
        });
        
        addSafeListener('normal-input', 'change', function() {
          Controls.handleTextureUpload(this, 'normal', app);
        });
        
        addSafeListener('roughness-input', 'change', function() {
          Controls.handleTextureUpload(this, 'roughness', app);
        });
        
        // Texture removal listeners
        addSafeListener('remove-albedo', 'click', function() {
          Controls.removeTexture('albedo', app);
        });
        
        addSafeListener('remove-normal', 'click', function() {
          Controls.removeTexture('normal', app);
        });
        
        addSafeListener('remove-roughness', 'click', function() {
          Controls.removeTexture('roughness', app);
        });
        
        // Mesh type selection
        addSafeListener('mesh-type', 'change', function() {
          app.options.currentMeshType = this.value;
          app.meshes = Models.updateMeshType(app.scene, app.meshes, app.material, this.value);
        });

        addSafeListener('preset-wood', 'click', function() {
          Materials.applyPreset(app.material, 'wood');
        });

        addSafeListener('preset-metal', 'click', function() {
          Materials.applyPreset(app.material, 'metal');
        });
        
        
        // (x to reset view)
        document.addEventListener('keydown', function(event) {
          if (event.key === 'x' || event.key === 'X') {
            // Reset camera position
            app.camera.position.set(0, 0, 8);
            app.camera.lookAt(0, 0, 0);
            //reset object rotation
            for (let i = 0; i < App.meshes.length; i++) {
                App.rotation.x = 0;
                App.rotation.y = 0;
              }
            // console.log('View reset to default position');
          }
        });
      },
      removeTexture(textureType, app) {
        // 1. Set app texture to null
        app.textures[textureType] = null;
        
        // 2. Update the material
        Materials.updateTexture(app.material, textureType, null);
        
        // 3. Clear the preview image
        const previewId = `${textureType}-preview`;
        const previewElement = document.getElementById(previewId);
        if (previewElement) {
          previewElement.style.backgroundImage = '';
          previewElement.style.backgroundSize = '';
        }
        
        // 4. Reset the file input
        const inputId = `${textureType}-input`;
        const inputElement = document.getElementById(inputId);
        if (inputElement) {
          inputElement.value = '';
        }
        
        // 5. Disable the remove button
        const removeButtonId = `remove-${textureType}`;
        const removeButton = document.getElementById(removeButtonId);
        if (removeButton) {
          removeButton.disabled = true;
        }
      },

    handleTextureUpload(inputElement, textureType, app) {
      const file = inputElement.files[0];
      if (!file) return;

      const reader = new FileReader();
      
      reader.onload = function(e) {
        const textureLoader = new THREE.TextureLoader();
        const texture = textureLoader.load(e.target.result);
        
        // Update the preview
        const previewId = inputElement.id.replace('-input', '-preview');
        document.getElementById(previewId).style.backgroundImage = `url(${e.target.result})`;
        document.getElementById(previewId).style.backgroundSize = 'cover';
        
        // Enable remove button
        const removeButtonId = `remove-${textureType}`;
        document.getElementById(removeButtonId).disabled = false;
        
        // Apply texture to material
        app.textures[textureType] = texture;
        Materials.updateTexture(app.material, textureType, texture);
      };
      
      reader.readAsDataURL(file);
    },

    createOrbitControls(camera, domElement) {
        return new THREE.OrbitControls(camera, domElement);
    }
  };