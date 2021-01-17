function cell_removed = XxRemoveCellItems(cell_list_raw, items_to_remove)

% ------------------------------------------------------------------------
% XxRemoveCellItems: remove specific items from a cell list
%
% usage:  cell_removed = XxRemoveCellItems(cell_raw, items_to_remove)
% where,
%    cell_raw          -- raw cell list
%    items_to_remove   -- items to be removed from cell_list_raw
%
% Author: Chang Qiao
% Email: qc17@mails.tsinghua.edu.cn
% Version: 2019/4/4
% ------------------------------------------------------------------------

nitems = max(size(cell_list_raw));
n_to_remove = max(size(items_to_remove));
flag = ones(size(cell_list_raw));
for i = 1:nitems
    cur_item = cell_list_raw{i};
    for j = 1:n_to_remove
        if strcmp(cur_item,items_to_remove{j})
            flag(i) = 0;
            break;
        end
    end
end
cell_removed = cell_list_raw(logical(flag));