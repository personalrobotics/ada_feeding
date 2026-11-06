#!/bin/bash
# Copyright (c) 2026, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

# Smoke test (#221): builds the Docker image from this checkout, then runs one
# simulated feeding session in dummy mode, driven through the web app. Needs
# only Docker. Exits 0 if the session passed.
#
#   smoke_test/run.sh [artifacts_dir]
#
# Artifacts go to smoke_test/artifacts/<UTC timestamp>/ by default.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
image="${IMAGE:-articutool-dev}"
artifacts="${1:-$here/artifacts/$(date -u +%Y%m%dT%H%M%SZ)}"
container="ada-smoke-$$"
# Keep ROS 2 traffic on the container's loopback interface (see README).
cyclonedds='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" multicast="true"/></Interfaces></General></Domain></CycloneDDS>'

mkdir -p "$artifacts"
{
    echo "ada_feeding commit: $(git -C "$repo" rev-parse HEAD 2> /dev/null || echo unknown)"
    echo "uncommitted changes: $(git -C "$repo" status --porcelain 2> /dev/null | wc -l | tr -d ' ') files"
    echo "host: $(uname -srm)"
    echo "docker: $(docker version --format '{{.Server.Version}} ({{.Server.Os}}/{{.Server.Arch}})')"
} > "$artifacts/host.txt"

echo "Building $image (log: $artifacts/docker_build.log)"
if ! docker build -t "$image" "$repo" > "$artifacts/docker_build.log" 2>&1; then
    echo "FAIL: docker build failed (see $artifacts/docker_build.log)"
    exit 1
fi
echo "image: $(docker image inspect --format '{{.Id}}' "$image")" >> "$artifacts/host.txt"

docker run -d --name "$container" -e CYCLONEDDS_URI="$cyclonedds" "$image" sleep infinity > /dev/null
cleanup() {
    docker cp "$container:/tmp/artifacts/." "$artifacts" > /dev/null 2>&1 || true
    docker rm -f "$container" > /dev/null
}
trap cleanup EXIT

docker exec "$container" mkdir -p /tmp/smoke_test
docker cp "$here/session.sh" "$container:/tmp/smoke_test/"
docker cp "$here/meal.js" "$container:/tmp/smoke_test/"
status=0
docker exec "$container" /tmp/smoke_test/session.sh /tmp/artifacts | tee "$artifacts/session.log" || status=$?
echo "Artifacts: $artifacts"
exit $status
