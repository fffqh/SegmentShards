import copy
import torch
import random
from random import randint
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

from typing import Any, Dict, List, Optional, Tuple
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
from torchvision.transforms.functional import resize, to_pil_image 

from .modeling import Sam
from .utils.transforms import ResizeLongestSide
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

class SamRPBATargetAdversarialGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0,#0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512/1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        return_logists:bool = True,
        adv_e:float = 0.05,
        adv_epsilon:float = 0.4,
        adv_iter:int = 100,
    ) -> None:
        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        #self.predictor = SamPredictor(model)
        self.sam = model
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.return_logists = return_logists
        
        self.adv_e = adv_e
        self.adv_eps = adv_epsilon
        self.adv_iter = adv_iter
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
    def _set_target(
        self,
        image:np.ndarray,
        image_format:str = "RGB",
    )->torch.Tensor:
        # check for image format
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.sam.image_format:
            image = image[..., ::-1]#最后一维作反转
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        return self.sam.preprocess(input_image_torch)
        
    def run(
            self,
            image:np.ndarray,
            target_image:np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        
        random.seed(1024)

        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        adv_image = copy.deepcopy(image)
        # process_crop
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            x0, y0, x1, y1 = crop_box
            cropped_im = image[y0:y1, x0:x1, :]
            cropped_im_target = target_image[y0:y1, x0:x1, :]
            cropped_im_size = cropped_im.shape[:2]

            # Transform the image to the form expected by the model
            im_torch_ori = self._set_image(cropped_im)
            im_torch_target = self._set_target(cropped_im_target)

            # Get points for this crop
            points_scale = np.array(cropped_im_size)[None, ::-1]
            points_for_image = self.point_grids[layer_idx] * points_scale
            
            im_torch = copy.deepcopy(im_torch_ori)
            self.tg_low_res_masks = None
            self.tg_iou_predictions = None
            # process_batch
            bm = torch.zeros_like(im_torch).detach().to(im_torch.device)
            
            batch_datas = []
            for j, item in enumerate(batch_iterator(self.points_per_batch, points_for_image)):
                batch_datas.append((j, item))
            N_batch = len(batch_datas)
            with tqdm(range(self.adv_iter)) as tbar:
                for i in tbar:
                    #random.shuffle(batch_datas)
                    idx = randint(0, N_batch-1)
                    #for (j,(points,)) in batch_datas:
                    ## points数据处理
                    j, (points,) = batch_datas[idx]
                    points_torch_tuple = self._set_points(points, cropped_im_size)
                    ## 进行攻击
                    iou_preds, _, adv_img_torch, bm = self._attack(im_torch, im_torch_ori, im_torch_target, points_torch_tuple, 
                                                                            bm, tbar, i,j)
                    im_torch = adv_img_torch
                
                    # 用im_torch更新adv_image
                    #im_torch = im_torch*self.sam.pixel_std + self.sam.pixel_mean
                    im_array = self._get_advimage(im_torch)
                    adv_image[y0:y1,x0:x1,:] = im_array
        return adv_image
    
    def _forward_sam(
        self,
        im_torch:torch.Tensor,
        points_torch:Tuple[torch.Tensor, torch.Tensor],
    )->Tuple[torch.Tensor, torch.Tensor]:
        # 计算 image_embeddings
        img_embeddings = self.sam.image_encoder(im_torch)
        # 计算 prompt_embeddings
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points = points_torch,
            boxes=None,
            masks=None
        )
        # 计算decoder输出
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=img_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        return low_res_masks, iou_predictions
    
    def _attack(
        self,
        im_torch_ini:torch.Tensor,
        im_torch_ori:torch.Tensor,
        im_torch_target:torch.Tensor,
        points_torch:Tuple[torch.Tensor, torch.Tensor],
        bm:torch.Tensor,
        tbar:Any,
        i:int,
        j:int,
    )-> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        
        # print("test im_ori:", im_torch_ori.max(), im_torch_ori.min())
        # print("test im_ini:", im_torch_ini.max(), im_torch_ini.min())
        # print("test im_tgt:", im_torch_target.max(), im_torch_target.min())
        
        # get target
        tg_low_res_masks, tg_iou_predictions = self._forward_sam(im_torch_target, points_torch)
        self.tg_low_res_masks = tg_low_res_masks.detach()
        self.tg_iou_predictions = tg_iou_predictions.detach()
        del tg_low_res_masks
        del tg_iou_predictions

        im_torch = Variable(im_torch_ini, requires_grad=True)
        optimizer = torch.optim.Adam([im_torch])
        loss_function = torch.nn.MSELoss()
        momentum = bm

        low_res_masks, iou_predictions = self._forward_sam(im_torch, points_torch)

        # 计算攻击损失函数值
        loss_mask = loss_function(low_res_masks, self.tg_low_res_masks)
        loss_iou = loss_function(iou_predictions, self.tg_iou_predictions)
        loss = loss_mask + loss_iou
        optimizer.zero_grad()
        loss.backward()

        tbar.set_postfix(epoch=i, 
                        loss='{0:1.4f}:({0:1.4f} {0:1.4f})'.format(
                            loss.data.item(),
                            loss_mask.data.item(),
                            loss_iou.data.item()))

        # 计算梯度
        grad = im_torch.grad.detach()
        grad = 0.9 * momentum + grad / (grad.abs().sum() + 1e-8)
        momentum = grad
        # 更新图像
        im_torch.data = im_torch.data - self.adv_e*torch.sign(grad)
        im_torch.data = torch.max(torch.min(im_torch.data, im_torch_ori+self.adv_eps), 
                                    im_torch_ori-self.adv_eps)
        
        iou_predictions = iou_predictions.detach().cpu()
        low_res_masks = low_res_masks.detach().cpu()
        im_torch = im_torch.detach()
        momentum = momentum.detach()
        return iou_predictions, low_res_masks, im_torch, momentum

    def _get_advimage(
        self,
        im_torch:torch.Tensor,
    )-> np.ndarray:
        im_torch = im_torch.detach().cpu().squeeze(0).numpy().transpose(1,2,0)
        im_max = im_torch.max()
        im_min = im_torch.min()
        im_torch = (im_torch - im_min) / (im_max - im_min) * 255.0
        im_torch = im_torch.astype(np.uint8)
        #print("im_torch shape:", im_torch.shape)
        #print("input_size:", self.input_size)
        im_torch = im_torch[0:self.input_size[0], 0:self.input_size[1], :]
        #print("debug im_torch shape:", im_torch.shape, im_torch.max(), im_torch.min())
        im_array = np.array(resize(to_pil_image(im_torch),self.original_size))
        return im_array

    def _maskdata_postprocess(
        self,
        mask_data:MaskData,
    )-> List[Dict[str, Any]]:
        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _cropdata_postprocess(
        self,
        data:MaskData,
        crop_box: List[int],
    )->MaskData:
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:,0]),
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
        return data

    def _batchdata_postprocess(
        self,
        data:MaskData,
        orig_size: Tuple[int, ...],
        crop_box: List[int],
    )-> MaskData:
        orig_h, orig_w = orig_size
        # 1. 根据阈值筛选
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)
        # 2. Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.sam.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)
        # 3. Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.sam.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])                
        
        # 4. Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)
        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]
        return data


    def _set_points(
        self,
        points:np.ndarray,
        im_size:Tuple[int, ...]
    )->Tuple[torch.Tensor,torch.Tensor]:
        transformed_points = self.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        
        return (in_points[:, None, :], in_labels[:, None])

    def _set_image(
        self,
        image:np.ndarray,
        image_format:str = "RGB",
    ) -> torch.Tensor:
        # check for image format
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.sam.image_format:
            image = image[..., ::-1]#最后一维作反转
        
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        self.original_size = image.shape[:2]
        self.input_size = tuple(input_image_torch.shape[-2:])
        return self.sam.preprocess(input_image_torch)

    @property
    def device(self) -> torch.device:
        return self.sam.device

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

