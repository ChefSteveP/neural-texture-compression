// import * as THREE from 'three';

const Renderer = {
    createScene() {  // Blank Scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x222222);
      return scene;
    },
    
    createCamera() { // Camera (fov, aspect, near, far)
      const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.z = 8;
      return camera;
    },
    
    createRenderer() {
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.shadowMap.enabled = true;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.0;
      document.getElementById('canvas-container').appendChild(renderer.domElement);
      return renderer;
    },
    
    setupLights(scene) {
      // Ambient light
      const ambient = new THREE.AmbientLight(0xffffff, 0.3);
      scene.add(ambient);
      
      // Directional light
      const directional = new THREE.DirectionalLight(0xffffff, 0.8);
      directional.position.set(1, 1, 1);
      directional.castShadow = true;
      scene.add(directional);
      
      // Point lights
      const point1 = new THREE.PointLight(0xffffff, 0.5);
      point1.position.set(2, 2, 2);
      scene.add(point1);

      
    //   const point2 = new THREE.PointLight(0xffffff, 0.3);
    //   point2.position.set(-2, -1, -1);
    //   scene.add(point2);
    },
    
    onWindowResize(camera, renderer) {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }
  };