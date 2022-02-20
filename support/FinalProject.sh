FILES="samples/Demo/dog_0100.jpg
samples/Demo/Without.jpg
samples/Demo/blank.jpeg
samples/Demo/Partial.jpg
samples/Demo/Masked.jpg
samples/Demo/twoWomen.jpeg"
for f in $FILES
do

echo "$f"
imagej $f
cd ~/Project/ && time python src/FinalProject.py -H out/CombinedModel.pth -M out/MaskModel.pth -N out/NaturalModel.pth -F $f
done
