CB Nano is a R&D startup that aims to develop new material and solutions to improve heat dissipation. The vision is to make data centers as climate friendly as possible. Our mission is to challenge what's possible from the atom up - combining nano material engineering with PUE-focused solutions.

## Existing Hardware
Our R&D computers setup are as follows. All of these machines can be repurposed to assume new roles should the need arise:

- 'ae86': an aging MacPro 5,1 running Debian 12 with still very capable dual Xeon processor, 128GB RAM, a 480GB SSD for the OS and a ZFS-backed RAID 5 with a 8TB pool. Its purpose is to be the company's Docker machine and NAS. It's suitable for any enterprise tasks that does not require a GPU. It's full details are in the file `inventory-ae86.json`.

- 'atom': a Mac Mini M4 Pro with 64GB RAM. Currently a developer's machine.

- 'lab': a Raspberry Pi 4B with 8GB. It is running our custom software called 'Labrador', a lab automation software that allows users to create test sequences to control lab equipments to perform reliable and repeatable experiments. The results are also stored at `results.marina.cbnano.com`, which is one of the docker containers on 'ae86'. The full details of 'lab' are in the file `inventory-lab.json`.

- 'pegasus': a developer's machine with 32GB RAM and a Nvidia 5060 Ti 16GB. It's purpose is for code development, small scale AI R&D and general computing for the developer. `inventory-pegasus.json`.

- 'zima': a 4GB Zima Board running Debian 13. It's sole purpose is to run Nginx Proxy Manager and provide SSL reverse proxy service for the LAN. `inventory-zima.json`.

- The router is a Ubiquity Dream Machine Pro. VPN is setup using Wireguard.


## Purchased, but not yet deployed Hardware

- ASUS ROG Astral RTC 5090 32GB GPU
- MSI MAG B860 Tomahawk Wifi motherboard with Intel Core Ultra 7 265k, 64GB DDR5 RAM and 1TB NVMe
