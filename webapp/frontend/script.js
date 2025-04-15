async function uploadImage() {
    const input = document.getElementById("imageInput");
    const status = document.getElementById("status");
    const viewer = document.getElementById("viewer");
  
    if (!input.files.length) return alert("Upload an image first.");
  
    status.innerText = "Processing...";
    viewer.innerHTML = "";
  
    const formData = new FormData();
    formData.append("image", input.files[0]);
  
    try {
      const response = await fetch("http://localhost:5000/convert", {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) throw new Error("Server error");
  
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      status.innerText = "Model ready!";
  
      loadGLBModel(url);
    } catch (err) {
      console.error(err);
      status.innerText = "Error: Failed to convert image.";
    }
  }
  
  function loadGLBModel(url) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(500, 500);
  
    const viewer = document.getElementById("viewer");
    viewer.innerHTML = '';
    viewer.appendChild(renderer.domElement);
  
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(0, 1, 1).normalize();
    scene.add(light);
  
    const loader = new THREE.GLTFLoader();
    loader.load(url, function (gltf) {
      const model = gltf.scene;
      scene.add(model);
      camera.position.z = 2;
  
      const animate = function () {
        requestAnimationFrame(animate);
        model.rotation.y += 0.01;
        renderer.render(scene, camera);
      };
      animate();
    });
  }
  