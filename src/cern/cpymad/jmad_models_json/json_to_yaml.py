
import json,yaml,os



if __name__ == "__main__":
    folder='../_models/'
    for f in os.listdir(folder):
        if f[-12:]=='.cpymad.json':
            print("Converting "+f[:-12])
            d=json.load(file(folder+f))
            yaml.safe_dump(d,stream=file(folder+f[:-4]+'yml','w'))
