const App = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    meshes: [],
    material: null,
    textures: {
      albedo: null,
      normal: null,
      roughness: null
    },
    options: {
      currentMeshType: 'sphere',
      syncRotation: true
    }, 
    mouse: {
      isDown: false,
      lastX: 0,
      lastY: 0,
      sensitivity: 0.005
    }, 
    rotation: {
        x: 0, 
        y: 0
    }
  };

  
  function init() {
    
    App.scene = Renderer.createScene();      // Create scene
    App.camera = Renderer.createCamera();    // Create camera
    App.renderer = Renderer.createRenderer();// Create renderer 

    //setup Lights
    Renderer.setupLights(App.scene);
    
    App.material = Materials.createDefaultMaterial();

    App.meshes = Models.createMeshes(App.scene, App.material, App.options.currentMeshType);
    
    // App.controls = Controls.createOrbitControls(App.camera, App.renderer.domElement);

    Controls.setupEventListeners(App);
    // Controls.setupKeyboardControls(App);
    Controls.setupMouseControls(App);

    window.addEventListener('resize', () => Renderer.onWindowResize(App.camera, App.renderer));

    animate();
  }

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        
        for (let i = 0; i < App.meshes.length; i++) {

          // Reset to identity quaternion first
          App.meshes[i].quaternion.set(0, 0, 0, 1);
          
          // Create rotation quaternions for each axis separately
          const euler = new THREE.Euler(App.rotation.x, App.rotation.y, 0, 'XYZ');
          const quaternion = new THREE.Quaternion().setFromEuler(euler);
          
          // Apply quaternion directly
          App.meshes[i].quaternion.copy(quaternion);
        }

        
        App.renderer.render(App.scene, App.camera);
    }
  
  window.addEventListener('load', init);