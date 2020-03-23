from Bio import SeqIO
from Bio import Entrez
import os
import time


base_folder = "./data/genbank/"

Entrez.email = 'pellegre.esteban@gmail.com'

viprbrc_data = {"coronaviridae": "./data/viprbrc/463186397445-CoronaviridaeGenomicFastaResults.fasta",
                "caliciviridae": "./data/viprbrc/36510026494-CaliciviridaeGenomicFastaResults.fasta",
                "hepeviridae": "./data/viprbrc/550458686357-HepeviridaeGenomicFastaResults.fasta",
                "picornavirida": "./data/viprbrc/770566255833-PicornaviridaeGenomicFastaResults.fasta",
                "togaviridae": "./data/viprbrc/77809854226-TogaviridaeGenomicFastaResults.fasta"}

bigd_data = {"coronaviridae": "./data/bigd/Selected_GWH_Coronavirus_Sequences_2020-3-23-21-1-45.fasta"}

genome_files = viprbrc_data
viruses = viprbrc_data.keys()


def fetch_gb(path, gid):
    handle = Entrez.efetch(db='nucleotide', id=gid, rettype='gb')
    local_file = open(path + "/" + gid + ".gb", 'w')
    local_file.write(handle.read())
    handle.close()
    local_file.close()


def main():
    print("[+] fetch genbank genomes")

    genome_files = viprbrc_data
    viruses = viprbrc_data.keys()

    for vrs in viruses:
        print("[+] reading records of ", vrs)

        data_folder = base_folder + "/" + vrs
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        for seq_record in SeqIO.parse(genome_files[vrs], "fasta"):
            genbank_id = seq_record.id.split(":")[1].split("|")[0]
            file_name = data_folder + "/" + genbank_id + ".gb"

            print("[+]  --- * fetching (", vrs, ")", genbank_id)

            if not os.path.exists(file_name):
                downloaded = False
                while not downloaded:
                    try:
                        fetch_gb(data_folder, genbank_id)
                        downloaded = True
                    except:
                        time.sleep(3)
                        pass
            else:
                print("[+]  (already downloaded) ")


main()
