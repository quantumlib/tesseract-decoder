# 1. Start a temporary background container named 'sysroot-builder'
docker run -d --name sysroot-builder debian:10 sleep 3600

# 2. Run the installation and packaging process inside the container
docker exec sysroot-builder bash -c "
  sed 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list  -i 

  apt-get update && apt-get upgrade -y && apt-get install -y build-essential libc6-dev symlinks apt-utils
  
  # Convert absolute symlinks to relative (CRITICAL so Clang doesn't read your host OS files)
  symlinks -rc /lib /usr/lib /usr/include
  
  # Create the staging directory
  mkdir -p /sysroot/usr/lib /sysroot/usr/include /sysroot/lib /sysroot/lib64
    
  ldd --version

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

# create custom_sysroot/BUILD
echo """filegroup(
    name = \"sysroot\",
    srcs = glob([\"**/*\"]),
    visibility = [\"//visibility:public\"],
)""" > custom_sysroot/BUILD

# update MODULE.bazel to use the custom toolchain
echo """bazel_dep(name = \"toolchains_llvm\", version = \"1.6.0\")

llvm = use_extension(\"@toolchains_llvm//toolchain/extensions:llvm.bzl\", \"llvm\")
llvm.toolchain(
    name = \"llvm_toolchain\",
    llvm_version = \"17.0.6\",
)

llvm.sysroot(
    name = \"llvm_toolchain\",
    label = \"//custom_sysroot:sysroot\",
    targets = [\"linux-x86_64\"],
)
use_repo(llvm, \"llvm_toolchain\")

register_toolchains(\"@llvm_toolchain//:all\")
""" >> MODULE.bazel


sudo apt-get update
sudo apt-get install -y libxml2

cd custom_sysroot/lib64/
rm -f ld-linux-x86-64.so.2
ln -s ../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 ld-linux-x86-64.so.2
cd ../..
