# -*- coding: utf-8 -*-
import traceback
import torch
from scipy.io import wavfile
import edge_tts
import subprocess
import gradio as gr
import gradio.processing_utils as gr_pu
import io
import os
import logging
import time
from pathlib import Path
import re
import json
import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")


logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def vc_fn(sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold):
    try:
        if input_audio is None:
            raise gr.Error("你需要上传音频")
        if model is None:
            raise gr.Error("你需要指定模型")
        sampling_rate, audio = input_audio
        # print(audio.shape,sampling_rate)
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        temp_path = "temp.wav"
        soundfile.write(temp_path, audio, sampling_rate, format="wav")
        _audio = model.slice_inference(temp_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale,
                                       pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold)
        model.clear_empty()
        os.remove(temp_path)
        # 构建保存文件的路径，并保存到results文件夹内
        try:
            timestamp = str(int(time.time()))
            filename = sid + "_" + timestamp + ".wav"
            # output_file = os.path.join("./results", filename)
            # soundfile.write(output_file, _audio, model.target_sample, format="wav")
            soundfile.write('/tmp/'+filename, _audio,
                            model.target_sample, format="wav")
            # return f"推理成功，音频文件保存为results/{filename}", (model.target_sample, _audio)
            return f"推理成功，音频文件保存为{filename}", (model.target_sample, _audio)
        except Exception as e:
            if debug:
                traceback.print_exc()
            return f"文件保存失败，请手动保存", (model.target_sample, _audio)
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def tts_func(_text, _rate, _voice):
    # 使用edge-tts把文字转成音频
    # voice = "zh-CN-XiaoyiNeural"#女性，较高音
    # voice = "zh-CN-YunxiNeural"#男性
    voice = "zh-CN-YunxiNeural"  # 男性
    if (_voice == "女"):
        voice = "zh-CN-XiaoyiNeural"
    output_file = "/tmp/"+_text[0:10]+".wav"
    # communicate = edge_tts.Communicate(_text, voice)
    # await communicate.save(output_file)
    if _rate >= 0:
        ratestr = "+{:.0%}".format(_rate)
    elif _rate < 0:
        ratestr = "{:.0%}".format(_rate)  # 减号自带

    p = subprocess.Popen("edge-tts " +
                         " --text "+_text +
                         " --write-media "+output_file +
                         " --voice "+voice +
                         " --rate="+ratestr, shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    p.wait()
    return output_file


def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)


def vc_fn2(sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, text2tts, tts_rate, tts_voice, f0_predictor, enhancer_adaptive_key, cr_threshold):
    # 使用edge-tts把文字转成音频
    text2tts = text_clear(text2tts)
    output_file = tts_func(text2tts, tts_rate, tts_voice)

    # 调整采样率
    sr2 = 44100
    wav, sr = librosa.load(output_file)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=sr2)
    save_path2 = text2tts[0:10]+"_44k"+".wav"
    wavfile.write(save_path2, sr2,
                  (wav2 * np.iinfo(np.int16).max).astype(np.int16)
                  )

    # 读取音频
    sample_rate, data = gr_pu.audio_from_file(save_path2)
    vc_input = (sample_rate, data)

    a, b = vc_fn(sid, vc_input, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale,
                 pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold)
    os.remove(output_file)
    os.remove(save_path2)
    return a, b


models_info = [
    {
        "description": """
                       这个模型包含碧蓝档案的141名角色。\n\n
                       Space采用CPU推理，速度极慢，建议下载模型本地GPU推理。\n\n
                       """,
        "model_path": "./G_228800.pth",
        "config_path": "./config.json",
    }
]

model_inferall = []
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        default=False, help="share gradio app")
    # 一定要设置的部分
    parser.add_argument('-cl', '--clip', type=float,
                        default=0, help='音频强制切片，默认0为自动切片，单位为秒/s')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+',
                        default=["君の知らない物語-src.wav"], help='wav文件名列表，放在raw文件夹下')
    parser.add_argument('-t', '--trans', type=int, nargs='+',
                        default=[0], help='音高调整，支持正负（半音）')
    parser.add_argument('-s', '--spk_list', type=str,
                        nargs='+', default=['nen'], help='合成目标说话人名称')

    # 可选项部分
    parser.add_argument('-a', '--auto_predict_f0', action='store_true',
                        default=False, help='语音转换自动预测音高，转换歌声时不要打开这个会严重跑调')
    parser.add_argument('-cm', '--cluster_model_path', type=str,
                        default="logs/44k/kmeans_10000.pt", help='聚类模型路径，如果没有训练聚类则随便填')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float,
                        default=0, help='聚类方案占比，范围0-1，若没有训练聚类模型则默认0即可')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0,
                        help='两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒')
    parser.add_argument('-f0p', '--f0_predictor', type=str, default="pm",
                        help='选择F0预测器,可选择crepe,pm,dio,harvest,默认为pm(注意：crepe为原F0使用均值滤波器)')
    parser.add_argument('-eh', '--enhance', action='store_true', default=False,
                        help='是否使用NSF_HIFIGAN增强器,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭')
    parser.add_argument('-shd', '--shallow_diffusion', action='store_true',
                        default=False, help='是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN增强器将会被禁止')

    # 浅扩散设置
    parser.add_argument('-dm', '--diffusion_model_path', type=str,
                        default="logs/44k/diffusion/model_0.pt", help='扩散模型路径')
    parser.add_argument('-dc', '--diffusion_config_path', type=str,
                        default="logs/44k/diffusion/config.yaml", help='扩散模型配置文件路径')
    parser.add_argument('-ks', '--k_step', type=int,
                        default=100, help='扩散步数，越大越接近扩散模型的结果，默认100')
    parser.add_argument('-od', '--only_diffusion', action='store_true',
                        default=False, help='纯扩散模式，该模式不会加载sovits模型，以扩散模型推理')

    # 不用动的部分
    parser.add_argument('-sd', '--slice_db', type=int,
                        default=-40, help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50')
    parser.add_argument('-d', '--device', type=str,
                        default=None, help='推理设备，None则为自动选择cpu和gpu')
    parser.add_argument('-ns', '--noice_scale', type=float,
                        default=0.4, help='噪音级别，会影响咬字和音质，较为玄学')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5,
                        help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现')
    parser.add_argument('-wf', '--wav_format', type=str,
                        default='flac', help='音频输出格式')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float,
                        default=0.75, help='自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭')
    parser.add_argument('-eak', '--enhancer_adaptive_key',
                        type=int, default=0, help='使增强器适应更高的音域(单位为半音数)|默认为0')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,
                        help='F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音')
    args = parser.parse_args()
    categories = ["Blue Archive"]
    others = {
        "PCR vits-fast-fineturning": "https://huggingface.co/spaces/FrankZxShen/vits-fast-finetuning-pcr",
        "Blue Archive vits-fast-fineturning": "https://huggingface.co/spaces/FrankZxShen/vits-fast-finetuning-ba",
    }
    for info in models_info:
        config_path = info['config_path']
        model_path = info['model_path']
        description = info['description']
        clean_names = args.clean_names
        trans = args.trans
        spk_list = list(get_hparams_from_file(config_path).spk.keys())
        slice_db = args.slice_db
        wav_format = args.wav_format
        auto_predict_f0 = args.auto_predict_f0
        cluster_infer_ratio = args.cluster_infer_ratio
        noice_scale = args.noice_scale
        pad_seconds = args.pad_seconds
        clip = args.clip
        lg = args.linear_gradient
        lgr = args.linear_gradient_retain
        f0p = args.f0_predictor
        enhance = args.enhance
        enhancer_adaptive_key = args.enhancer_adaptive_key
        cr_threshold = args.f0_filter_threshold
        diffusion_model_path = args.diffusion_model_path
        diffusion_config_path = args.diffusion_config_path
        k_step = args.k_step
        only_diffusion = args.only_diffusion
        shallow_diffusion = args.shallow_diffusion

        model = Svc(model_path, config_path, args.device, args.cluster_model_path, enhance,
                    diffusion_model_path, diffusion_config_path, shallow_diffusion, only_diffusion)

        model_inferall.append((description, spk_list, model))

    app = gr.Blocks()
    with app:
        gr.Markdown(
            "# <center> so-vits-svc-models-ba\n"
            "# <center> Pay attention！！！ Space uses CPU inferencing, which is extremely slow. It is recommended to download models.\n"
            "# <center> 注意！！！Space采用CPU推理，速度极慢，建议下载模型使用本地GPU推理。\n"
            "## <center> Please do not generate content that could infringe upon the rights or cause harm to individuals or organizations.\n"
            "## <center> 请不要生成会对个人以及组织造成侵害的内容\n\n"
        )
        gr.Markdown("# Blue Archive\n\n"
                    )
        with gr.Tabs():
            for category in categories:
                with gr.TabItem(category):
                    for i, (description, speakers, model) in enumerate(
                            model_inferall):
                        gr.Markdown(description)
                        with gr.Row():
                            with gr.Column():
                                # textbox = gr.TextArea(label="Text",
                                #                         placeholder="Type your sentence here ",
                                #                         value="新たなキャラを解放できるようになったようですね。", elem_id=f"tts-input")

                                gr.Markdown(value="""
                                    <font size=2> 推理设置</font>
                                    """)
                                sid = gr.Dropdown(
                                    choices=speakers, value=speakers[0], label='角色选择')
                                auto_f0 = gr.Checkbox(
                                    label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）", value=False)
                                f0_predictor = gr.Dropdown(label="选择F0预测器,可选择crepe,pm,dio,harvest,默认为pm(注意：crepe为原F0使用均值滤波器)", choices=[
                                                           "pm", "dio", "harvest", "crepe"], value="pm")
                                vc_transform = gr.Number(
                                    label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                                cluster_ratio = gr.Number(
                                    label="聚类模型混合比例，0-1之间，0即不启用聚类。使用聚类模型能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）", value=0)
                                slice_db = gr.Number(label="切片阈值", value=-40)
                                noise_scale = gr.Number(
                                    label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
                            with gr.Column():
                                pad_seconds = gr.Number(
                                    label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5)
                                cl_num = gr.Number(
                                    label="音频自动切片，0为不切片，单位为秒(s)", value=0)
                                lg_num = gr.Number(
                                    label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s", value=0)
                                lgr_num = gr.Number(
                                    label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭", value=0.75)
                                enhancer_adaptive_key = gr.Number(
                                    label="使增强器适应更高的音域(单位为半音数)|默认为0", value=0)
                                cr_threshold = gr.Number(
                                    label="F0过滤阈值，只有启动crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音", value=0.05)
                        with gr.Tabs():
                            with gr.TabItem("音频转音频"):
                                vc_input3 = gr.Audio(label="选择音频")
                                vc_submit = gr.Button(
                                    "音频转换", variant="primary")
                            with gr.TabItem("文字转音频"):
                                text2tts = gr.Textbox(
                                    label="在此输入要转译的文字。注意，使用该功能建议打开F0预测，不然会很怪")
                                tts_rate = gr.Number(label="tts语速", value=0)
                                tts_voice = gr.Radio(label="性别", choices=[
                                                     "男", "女"], value="男")
                                vc_submit2 = gr.Button(
                                    "文字转换", variant="primary")
                        with gr.Row():
                            with gr.Column():
                                vc_output1 = gr.Textbox(label="Output Message")
                            with gr.Column():
                                vc_output2 = gr.Audio(
                                    label="Output Audio", interactive=False)

                vc_submit.click(vc_fn, [sid, vc_input3, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,
                                        cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold], [vc_output1, vc_output2])
                vc_submit2.click(vc_fn2, [sid, vc_input3, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num,
                                          lg_num, lgr_num, text2tts, tts_rate, tts_voice, f0_predictor, enhancer_adaptive_key, cr_threshold], [vc_output1, vc_output2])
                # gr.Examples(
                #     examples=example,
                #     inputs=[textbox, char_dropdown, language_dropdown,
                #             duration_slider, symbol_input],
                #     outputs=[text_output, audio_output],
                #     fn=tts_fn
                # )
            for category, link in others.items():
                with gr.TabItem(category):
                    gr.Markdown(
                        f'''
                        <center>
                          <h2>Click to Go</h2>
                          <a href="{link}">
                            <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-xl-dark.svg"
                          </a>
                        </center>
                        '''
                    )

    app.queue(concurrency_count=3).launch(show_api=False, share=args.share)
