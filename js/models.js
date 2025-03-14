const Models = {
    createMeshes(scene, material, type) {
      const meshes = [];
      const geometry = this.createGeometry(type);
      
      // Create three meshes with same material but different positions
      for (let i = 0; i < 3; i++) {
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.x = (i - 1) * 4; // Position them side by side
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
        meshes.push(mesh);
      }
      
      return meshes;
    },
    
    createGeometry(type) {
      switch (type) {
        case 'sphere':
          return new THREE.SphereGeometry(1.5, 64, 64);
        case 'cube':
          return new THREE.BoxGeometry(2, 2, 2, 10, 10, 10);
        case 'cylinder':
          return new THREE.CylinderGeometry(1, 1, 3, 32);
        case 'torus':
          return new THREE.TorusGeometry(1.0, 0.35, 16, 100);
        case 'torusKnot':
          return new THREE.TorusKnotGeometry(1.0, 0.4, 100, 16);
        case 'plane':
          return new THREE.PlaneGeometry(3.5, 3.5, 10, 10);
        default:
          return new THREE.SphereGeometry(1.5, 64, 64);
      }
    },
    
    syncMeshRotations(meshes) {
      for (let i = 1; i < meshes.length; i++) {
        meshes[i].rotation.copy(meshes[0].rotation);
      }
    },
    
    updateMeshType(scene, meshes, material, type) {
      // Remove existing meshes
      meshes.forEach(mesh => scene.remove(mesh));
      
      // Create new meshes
      return this.createMeshes(scene, material, type);
    }
  };