const Materials = {
    createDefaultMaterial() {
      return new THREE.MeshStandardMaterial({
        color: 0xcccccc,
        roughness: 0.5,
        metalness: 0.0,
      });
    },
    
    applyPreset(material, presetName) {
      switch (presetName) {
        case 'metal':
          material.map = new THREE.TextureLoader().load('sample_textures/metal-albedo.jpg');
          material.normalMap = new THREE.TextureLoader().load('sample_textures/metal-normal.jpg');
          material.roughnessMap = new THREE.TextureLoader().load('sample_textures/metal-roughness.jpg');
          material.metalness = 0.9;
          break;

        case 'wood':
          material.map = new THREE.TextureLoader().load('sample_textures/wood-albedo.jpg');
          material.normalMap = new THREE.TextureLoader().load('sample_textures/wood-normal.jpg');
          material.roughnessMap = new THREE.TextureLoader().load('sample_textures/wood-roughness.jpg');
          material.metalness = 0.0;
          break;
      }
      
      material.needsUpdate = true;
      return material;
    },
    
    updateTexture(material, textureType, textureMap) {
      switch (textureType) {

        case 'albedo':
          if (textureMap) {
            material.map = textureMap;
          } else { // default color
            material.color.set(0xcccccc);
            material.map = null;
          }
          break;

        case 'normal':
          if (textureMap) {
            material.normalMap = textureMap;
          } else {
            material.normalMap = null;
          }
          material.normalScale.set(1.0, 1.0);
          break;

        case 'roughness':
          if (textureMap == null) {
            material.roughnessMap = null;
            material.roughness= 0.5;
            break;
          }
          material.roughnessMap = textureMap;
          break;
      }
      
      material.needsUpdate = true;
      return material;
    }
  };