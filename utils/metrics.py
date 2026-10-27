import numpy as np
import torch


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan, mask_val=np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0, mask_val=np.nan) -> torch.Tensor:
    """Masked mean absolute percentage error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value.
                                    In the mape metric, null_val is set to 0.0 by all default.
                                    We keep this parameter for consistency, but we do not allow it to be changed.
                                    Zeros in labels will lead to inf in mape. Therefore, null_val is set to 0.0 by default.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    labels = torch.where(torch.abs(labels) < 1e-4, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.abs(preds-labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan, mask_val=np.nan) -> torch.Tensor:
    """Masked mean squared error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean squared error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan, mask_val=np.nan) -> torch.Tensor:
    """root mean squared error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value . Defaults to np.nan.

    Returns:
        torch.Tensor: root mean squared error
    """

    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, mask_val=mask_val))

def masked_mae_np(preds: np.ndarray, labels: np.ndarray, null_val: float = np.nan, mask_val=np.nan) -> float:
    """Masked mean absolute error.

    Args:
        preds (np.ndarray): predicted values
        labels (np.ndarray): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        float: masked mean absolute error
    """
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        eps = 5e-5
        mask = ~np.isclose(labels, null_val, atol=eps, rtol=0.0)
    if not np.isnan(mask_val):
        mask &= labels >= mask_val
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), 0, mask)
    loss = np.abs(preds - labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), 0, loss)
    return np.mean(loss)

def masked_mape_np(preds: np.ndarray, labels: np.ndarray, null_val: float = 0.0, mask_val=np.nan) -> float:
    """Masked mean absolute percentage error.

    Args:
        preds (np.ndarray): predicted values
        labels (np.ndarray): labels
        null_val (float, optional): null value. Defaults to 0.0.

    Returns:
        float: masked mean absolute percentage error
    """
    null_val = 0.0
    labels = np.where(np.abs(labels) < 1e-4, np.nan, labels)  # Mask near-zero values to prevent division by zero
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        eps = 5e-5
        mask = ~np.isclose(labels, null_val, atol=eps, rtol=0.0)
    if not np.isnan(mask_val):
        mask &= labels >= mask_val
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), 0, mask)
    loss = np.abs((preds - labels) / labels)
    loss = np.nan_to_num(loss)  # Replace NaN or infinity values with zeros
    loss = loss * mask
    return np.mean(loss)

def masked_mse_np(preds: np.ndarray, labels: np.ndarray, null_val: float = np.nan, mask_val=np.nan) -> float:
    """Masked mean squared error.

    Args:
        preds (np.ndarray): predicted values
        labels (np.ndarray): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        float: masked mean squared error
    """
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        eps = 5e-5
        mask = ~np.isclose(labels, null_val, atol=eps, rtol=0.0)
    if not np.isnan(mask_val):
        mask &= labels >= mask_val
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), 0, mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = np.where(np.isnan(loss), 0, loss)
    return np.mean(loss)

def masked_rmse_np(preds: np.ndarray, labels: np.ndarray, null_val: float = np.nan, mask_val=np.nan) -> float:
    """Root mean squared error.

    Args:
        preds (np.ndarray): predicted values
        labels (np.ndarray): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        float: root mean squared error
    """
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val, mask_val=mask_val))

def metric_numpy(pred: np.ndarray, true: np.ndarray, mask_val=np.nan):
    mae = masked_mae_np(pred, true, null_val=0.0, mask_val=mask_val)
    mse = masked_mse_np(pred, true, null_val=0.0, mask_val=mask_val)
    mape = masked_mape_np(pred, true, null_val=0.0, mask_val=mask_val)

    return mae, mse, mape

def metric_tensor(pred, true, mask_val=np.nan):
    mae = masked_mae(pred, true, null_val=0.0, mask_val=mask_val)
    mse = masked_mse(pred, true, null_val=0.0, mask_val=mask_val)
    mape = masked_mape(pred, true, null_val=0.0, mask_val=mask_val)

    return mae.item(), mse.item(), mape.item()
