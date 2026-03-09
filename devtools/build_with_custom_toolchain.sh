# 1. Start a temporary background container named 'sysroot-builder'
docker run -d --name sysroot-builder debian:11.11 sleep 3600

# 2. Run the installation and packaging process inside the container
docker exec sysroot-builder bash -c "
  apt-get update && apt-get install -y build-essential libc6-dev symlinks
  
  # Convert absolute symlinks to relative (CRITICAL so Clang doesn't read your host OS files)
  symlinks -rc /lib /usr/lib /usr/include
  
  # Create the staging directory
  mkdir -p /sysroot/usr/lib /sysroot/usr/include /sysroot/lib /sysroot/lib64
  
  # Copy files while preserving symlinks (-a)
  cp -a /usr/include/* /sysroot/usr/include/
  cp -a /usr/lib/* /sysroot/usr/lib/
  cp -a /lib/* /sysroot/lib/ 2>/dev/null || true
  cp -a /lib64/* /sysroot/lib64/ 2>/dev/null || true
  
  # Create a tarball inside the container
  cd /sysroot && tar -czf /sysroot.tar.gz .
"

# 3. Copy the tarball from the container directly to your host machine
docker cp sysroot-builder:/sysroot.tar.gz ./sysroot.tar.gz

# 4. Extract the contents into your project and clean up
mkdir -p custom_sysroot
tar -xzf sysroot.tar.gz -C custom_sysroot
rm sysroot.tar.gz
docker rm -f sysroot-builder

